# CC-LEARN: Cohort-based Consistency Learning 

**Authors**: Xiao Ye, Shaswat Shrivastava, Zhaonan Li, Jacob Dineen, Shijie Lu, Avneet Ahuja, Ming Shen, Zhikun Xu, Ben Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.15662)  

**Abstract**: Large language models excel at many tasks but still struggle with consistent, robust reasoning. We introduce Cohort-based Consistency Learning (CC-Learn), a reinforcement learning framework that improves the reliability of LLM reasoning by training on cohorts of similar questions derived from shared programmatic abstractions. To enforce cohort-level consistency, we define a composite objective combining cohort accuracy, a retrieval bonus for effective problem decomposition, and a rejection penalty for trivial or invalid lookups that reinforcement learning can directly optimize, unlike supervised fine-tuning. Optimizing this reward guides the model to adopt uniform reasoning patterns across all cohort members. Experiments on challenging reasoning benchmarks (including ARC-Challenge and StrategyQA) show that CC-Learn boosts both accuracy and reasoning stability over pretrained and SFT baselines. These results demonstrate that cohort-level RL effectively enhances reasoning consistency in LLMs. 

---
# The Compositional Architecture of Regret in Large Language Models 

**Authors**: Xiangxiang Cui, Shu Yang, Tianjin Huang, Wanyu Lin, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15617)  

**Abstract**: Regret in Large Language Models refers to their explicit regret expression when presented with evidence contradicting their previously generated misinformation. Studying the regret mechanism is crucial for enhancing model reliability and helps in revealing how cognition is coded in neural networks. To understand this mechanism, we need to first identify regret expressions in model outputs, then analyze their internal representation. This analysis requires examining the model's hidden states, where information processing occurs at the neuron level. However, this faces three key challenges: (1) the absence of specialized datasets capturing regret expressions, (2) the lack of metrics to find the optimal regret representation layer, and (3) the lack of metrics for identifying and analyzing regret neurons. Addressing these limitations, we propose: (1) a workflow for constructing a comprehensive regret dataset through strategically designed prompting scenarios, (2) the Supervised Compression-Decoupling Index (S-CDI) metric to identify optimal regret representation layers, and (3) the Regret Dominance Score (RDS) metric to identify regret neurons and the Group Impact Coefficient (GIC) to analyze activation patterns. Our experimental results successfully identified the optimal regret representation layer using the S-CDI metric, which significantly enhanced performance in probe classification experiments. Additionally, we discovered an M-shaped decoupling pattern across model layers, revealing how information processing alternates between coupling and decoupling phases. Through the RDS metric, we categorized neurons into three distinct functional groups: regret neurons, non-regret neurons, and dual neurons. 

---
# PhantomHunter: Detecting Unseen Privately-Tuned LLM-Generated Text via Family-Aware Learning 

**Authors**: Yuhui Shi, Yehan Yang, Qiang Sheng, Hao Mi, Beizhe Hu, Chaoxi Xu, Juan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2506.15683)  

**Abstract**: With the popularity of large language models (LLMs), undesirable societal problems like misinformation production and academic misconduct have been more severe, making LLM-generated text detection now of unprecedented importance. Although existing methods have made remarkable progress, a new challenge posed by text from privately tuned LLMs remains underexplored. Users could easily possess private LLMs by fine-tuning an open-source one with private corpora, resulting in a significant performance drop of existing detectors in practice. To address this issue, we propose PhantomHunter, an LLM-generated text detector specialized for detecting text from unseen, privately-tuned LLMs. Its family-aware learning framework captures family-level traits shared across the base models and their derivatives, instead of memorizing individual characteristics. Experiments on data from LLaMA, Gemma, and Mistral families show its superiority over 7 baselines and 3 industrial services, with F1 scores of over 96%. 

---
# Gender Inclusivity Fairness Index (GIFI): A Multilevel Framework for Evaluating Gender Diversity in Large Language Models 

**Authors**: Zhengyang Shan, Emily Ruth Diana, Jiawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.15568)  

**Abstract**: We present a comprehensive evaluation of gender fairness in large language models (LLMs), focusing on their ability to handle both binary and non-binary genders. While previous studies primarily focus on binary gender distinctions, we introduce the Gender Inclusivity Fairness Index (GIFI), a novel and comprehensive metric that quantifies the diverse gender inclusivity of LLMs. GIFI consists of a wide range of evaluations at different levels, from simply probing the model with respect to provided gender pronouns to testing various aspects of model generation and cognitive behaviors under different gender assumptions, revealing biases associated with varying gender identifiers. We conduct extensive evaluations with GIFI on 22 prominent open-source and proprietary LLMs of varying sizes and capabilities, discovering significant variations in LLMs' gender inclusivity. Our study highlights the importance of improving LLMs' inclusivity, providing a critical benchmark for future advancements in gender fairness in generative models. 

---
# PredGen: Accelerated Inference of Large Language Models through Input-Time Speculation for Real-Time Speech Interaction 

**Authors**: Shufan Li, Aditya Grover  

**Link**: [PDF](https://arxiv.org/pdf/2506.15556)  

**Abstract**: Large Language Models (LLMs) are widely used in real-time voice chat applications, typically in combination with text-to-speech (TTS) systems to generate audio responses. However, their large size often leads to noticeable latency between the end of user input and the start of audio output, resulting in suboptimal user experiences. This latency is particularly evident when LLMs are deployed as single-user voice assistants on consumer-grade hardware with limited computing capacity. We discovered that this latency is primarily dominated by the time it takes for the LLMs to generate the first sentence, which is required as input by the TTS systems that synthesize audio responses on a sentence-by-sentence basis. To address this bottleneck, we propose Predictive Generation (PredGen), a novel framework that mitigates-or even eliminates-this delay through speculative decoding at input time. PredGen generates candidate responses while the user is still speaking, enabling the system to begin TTS processing with minimal delay. Simulated experiments on the Lmsys and MT-Bench datasets show that the proposed method can effectively reduce the latency by around 2x across a wide range of use cases, while incurring only minimal additional computation cost at input time-computation that would otherwise go unused. 

---
# Lessons from Training Grounded LLMs with Verifiable Rewards 

**Authors**: Shang Hong Sim, Tej Deep Pala, Vernon Toh, Hai Leong Chieu, Amir Zadeh, Chuan Li, Navonil Majumder, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2506.15522)  

**Abstract**: Generating grounded and trustworthy responses remains a key challenge for large language models (LLMs). While retrieval-augmented generation (RAG) with citation-based grounding holds promise, instruction-tuned models frequently fail even in straightforward scenarios: missing explicitly stated answers, citing incorrectly, or refusing when evidence is available. In this work, we explore how reinforcement learning (RL) and internal reasoning can enhance grounding in LLMs. We use the GRPO (Group Relative Policy Optimization) method to train models using verifiable outcome-based rewards targeting answer correctness, citation sufficiency, and refusal quality, without requiring gold reasoning traces or expensive annotations. Through comprehensive experiments across ASQA, QAMPARI, ELI5, and ExpertQA we show that reasoning-augmented models significantly outperform instruction-only variants, especially in handling unanswerable queries and generating well-cited responses. A two-stage training setup, first optimizing answer and citation behavior and then refusal, further improves grounding by stabilizing the learning signal. Additionally, we revisit instruction tuning via GPT-4 distillation and find that combining it with GRPO enhances performance on long-form, generative QA tasks. Overall, our findings highlight the value of reasoning, stage-wise optimization, and outcome-driven RL for building more verifiable and reliable LLMs. 

---
# GenRecal: Generation after Recalibration from Large to Small Vision-Language Models 

**Authors**: Byung-Kwan Lee, Ryo Hachiuma, Yong Man Ro, Yu-Chiang Frank Wang, Yueh-Hua Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15681)  

**Abstract**: Recent advancements in vision-language models (VLMs) have leveraged large language models (LLMs) to achieve performance on par with closed-source systems like GPT-4V. However, deploying these models in real-world scenarios, particularly on resource-constrained devices, remains challenging due to their substantial computational demands. This has spurred interest in distilling knowledge from large VLMs into smaller, more efficient counterparts. A key challenge arises here from the diversity of VLM architectures, which are built on different LLMs and employ varying token types-differing in vocabulary size, token splits, and token index ordering. To address this challenge of limitation to a specific VLM type, we present Generation after Recalibration (GenRecal), a novel, general-purpose distillation framework for VLMs. GenRecal incorporates a Recalibrator that aligns and adapts feature representations between heterogeneous VLMs, enabling effective knowledge transfer across different types of VLMs. Through extensive experiments on multiple challenging benchmarks, we demonstrate that GenRecal significantly improves baseline performances, eventually outperforming large-scale open- and closed-source VLMs. 

---
# SPARE: Single-Pass Annotation with Reference-Guided Evaluation for Automatic Process Supervision and Reward Modelling 

**Authors**: Md Imbesat Hassan Rizvi, Xiaodan Zhu, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2506.15498)  

**Abstract**: Process or step-wise supervision has played a crucial role in advancing complex multi-step reasoning capabilities of Large Language Models (LLMs). However, efficient, high-quality automated process annotation remains a significant challenge. To address this, we introduce Single-Pass Annotation with Reference-Guided Evaluation (SPARE), a novel structured framework that enables single-pass, per-step annotation by aligning each solution step to one or multiple steps in a reference solution, accompanied by explicit reasoning for evaluation. We show that reference-guided step-level evaluation effectively facilitates process supervision on four datasets spanning three domains: mathematical reasoning, multi-hop compositional question answering, and spatial reasoning. We demonstrate that SPARE, when compared to baselines, improves reasoning performance when used for: (1) fine-tuning models in an offline RL setup for inference-time greedy-decoding, and (2) training reward models for ranking/aggregating multiple LLM-generated outputs. Additionally, SPARE achieves competitive performance on challenging mathematical datasets while offering 2.6 times greater efficiency, requiring only 38% of the runtime, compared to tree search-based automatic annotation. The codebase, along with a trained SPARE-PRM model, is publicly released to facilitate further research and reproducibility. 

---
# Context-Informed Grounding Supervision 

**Authors**: Hyunji Lee, Seunghyun Yoon, Yunjae Won, Hanseok Oh, Geewook Kim, Trung Bui, Franck Dernoncourt, Elias Stengel-Eskin, Mohit Bansal, Minjoon Seo  

**Link**: [PDF](https://arxiv.org/pdf/2506.15480)  

**Abstract**: Large language models (LLMs) are often supplemented with external knowledge to provide information not encoded in their parameters or to reduce hallucination. In such cases, we expect the model to generate responses by grounding its response in the provided external context. However, prior work has shown that simply appending context at inference time does not ensure grounded generation. To address this, we propose Context-INformed Grounding Supervision (CINGS), a post-training supervision in which the model is trained with relevant context prepended to the response, while computing the loss only over the response tokens and masking out the context. Our experiments demonstrate that models trained with CINGS exhibit stronger grounding in both textual and visual domains compared to standard instruction-tuned models. In the text domain, CINGS outperforms other training methods across 11 information-seeking datasets and is complementary to inference-time grounding techniques. In the vision-language domain, replacing a vision-language model's LLM backbone with a CINGS-trained model reduces hallucinations across four benchmarks and maintains factual consistency throughout the generated response. This improved grounding comes without degradation in general downstream performance. Finally, we analyze the mechanism underlying the enhanced grounding in CINGS and find that it induces a shift in the model's prior knowledge and behavior, implicitly encouraging greater reliance on the external context. 

---
# AgentGroupChat-V2: Divide-and-Conquer Is What LLM-Based Multi-Agent System Need 

**Authors**: Zhouhong Gu, Xiaoxuan Zhu, Yin Cai, Hao Shen, Xingzhou Chen, Qingyi Wang, Jialin Li, Xiaoran Shi, Haoran Guo, Wenxuan Huang, Hongwei Feng, Yanghua Xiao, Zheyu Ye, Yao Hu, Shaosheng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2506.15451)  

**Abstract**: Large language model based multi-agent systems have demonstrated significant potential in social simulation and complex task resolution domains. However, current frameworks face critical challenges in system architecture design, cross-domain generalizability, and performance guarantees, particularly as task complexity and number of agents increases. We introduces AgentGroupChat-V2, a novel framework addressing these challenges through three core innovations: (1) a divide-and-conquer fully parallel architecture that decomposes user queries into hierarchical task forest structures enabling dependency management and distributed concurrent processing. (2) an adaptive collaboration engine that dynamically selects heterogeneous LLM combinations and interaction modes based on task characteristics. (3) agent organization optimization strategies combining divide-and-conquer approaches for efficient problem decomposition. Extensive experiments demonstrate AgentGroupChat-V2's superior performance across diverse domains, achieving 91.50% accuracy on GSM8K (exceeding the best baseline by 5.6 percentage points), 30.4% accuracy on competition-level AIME (nearly doubling other methods), and 79.20% pass@1 on HumanEval. Performance advantages become increasingly pronounced with higher task difficulty, particularly on Level 5 MATH problems where improvements exceed 11 percentage points compared to state-of-the-art baselines. These results confirm that AgentGroupChat-V2 provides a comprehensive solution for building efficient, general-purpose LLM multi-agent systems with significant advantages in complex reasoning scenarios. Code is available at this https URL. 

---
# Revisiting Compositional Generalization Capability of Large Language Models Considering Instruction Following Ability 

**Authors**: Yusuke Sakai, Hidetaka Kamigaito, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.15629)  

**Abstract**: In generative commonsense reasoning tasks such as CommonGen, generative large language models (LLMs) compose sentences that include all given concepts. However, when focusing on instruction-following capabilities, if a prompt specifies a concept order, LLMs must generate sentences that adhere to the specified order. To address this, we propose Ordered CommonGen, a benchmark designed to evaluate the compositional generalization and instruction-following abilities of LLMs. This benchmark measures ordered coverage to assess whether concepts are generated in the specified order, enabling a simultaneous evaluation of both abilities. We conducted a comprehensive analysis using 36 LLMs and found that, while LLMs generally understand the intent of instructions, biases toward specific concept order patterns often lead to low-diversity outputs or identical results even when the concept order is altered. Moreover, even the most instruction-compliant LLM achieved only about 75% ordered coverage, highlighting the need for improvements in both instruction-following and compositional generalization capabilities. 

---
# DeVisE: Behavioral Testing of Medical Large Language Models 

**Authors**: Camila Zurdo Tagliabue, Heloisa Oss Boll, Aykut Erdem, Erkut Erdem, Iacer Calixto  

**Link**: [PDF](https://arxiv.org/pdf/2506.15339)  

**Abstract**: Large language models (LLMs) are increasingly used in clinical decision support, yet current evaluation methods often fail to distinguish genuine medical reasoning from superficial patterns. We introduce DeVisE (Demographics and Vital signs Evaluation), a behavioral testing framework for probing fine-grained clinical understanding. We construct a dataset of ICU discharge notes from MIMIC-IV, generating both raw (real-world) and template-based (synthetic) versions with controlled single-variable counterfactuals targeting demographic (age, gender, ethnicity) and vital sign attributes. We evaluate five LLMs spanning general-purpose and medically fine-tuned variants, under both zero-shot and fine-tuned settings. We assess model behavior via (1) input-level sensitivity - how counterfactuals alter the likelihood of a note; and (2) downstream reasoning - how they affect predicted hospital length-of-stay. Our results show that zero-shot models exhibit more coherent counterfactual reasoning patterns, while fine-tuned models tend to be more stable yet less responsive to clinically meaningful changes. Notably, demographic factors subtly but consistently influence outputs, emphasizing the importance of fairness-aware evaluation. This work highlights the utility of behavioral testing in exposing the reasoning strategies of clinical LLMs and informing the design of safer, more transparent medical AI systems. 

---
# Approximating Language Model Training Data from Weights 

**Authors**: John X. Morris, Junjie Oscar Yin, Woojeong Kim, Vitaly Shmatikov, Alexander M. Rush  

**Link**: [PDF](https://arxiv.org/pdf/2506.15553)  

**Abstract**: Modern language models often have open weights but closed training data. We formalize the problem of data approximation from model weights and propose several baselines and metrics. We develop a gradient-based approach that selects the highest-matching data from a large public text corpus and show its effectiveness at recovering useful data given only weights of the original and finetuned models. Even when none of the true training data is known, our method is able to locate a small subset of public Web documents can be used to train a model to close to the original model performance given models trained for both classification and supervised-finetuning. On the AG News classification task, our method improves performance from 65% (using randomly selected data) to 80%, approaching the expert benchmark of 88%. When applied to a model trained with SFT on MSMARCO web documents, our method reduces perplexity from 3.3 to 2.3, compared to an expert LLAMA model's perplexity of 2.0. 

---
# MinosEval: Distinguishing Factoid and Non-Factoid for Tailored Open-Ended QA Evaluation with LLMs 

**Authors**: Yongqi Fan, Yating Wang, Guandong Wang, Jie Zhai, Jingping Liu, Qi Ye, Tong Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2506.15215)  

**Abstract**: Open-ended question answering (QA) is a key task for evaluating the capabilities of large language models (LLMs). Compared to closed-ended QA, it demands longer answer statements, more nuanced reasoning processes, and diverse expressions, making refined and interpretable automatic evaluation both crucial and challenging. Traditional metrics like ROUGE and BERTScore struggle to capture semantic similarities due to different patterns between model responses and reference answers. Current LLM-based evaluation approaches, such as pairwise or listwise comparisons of candidate answers, lack intuitive interpretability. While pointwise scoring of each response provides some descriptions, it fails to adapt across different question contents. Most notably, existing methods overlook the distinction between factoid and non-factoid questions. To address these challenges, we propose \textbf{MinosEval}, a novel evaluation method that first distinguishes open-ended questions and then ranks candidate answers using different evaluation strategies. For factoid questions, it applies an adaptive key-point scoring strategy, while for non-factoid questions, it uses an instance-aware listwise ranking strategy. Experiments on multiple open-ended QA datasets, including self-built ones with more candidate responses to complement community resources, show that MinosEval better aligns with human annotations and offers more interpretable results. 

---
# TopClustRAG at SIGIR 2025 LiveRAG Challenge 

**Authors**: Juli Bakagianni, John Pavlopoulos, Aristidis Likas  

**Link**: [PDF](https://arxiv.org/pdf/2506.15246)  

**Abstract**: We present TopClustRAG, a retrieval-augmented generation (RAG) system developed for the LiveRAG Challenge, which evaluates end-to-end question answering over large-scale web corpora. Our system employs a hybrid retrieval strategy combining sparse and dense indices, followed by K-Means clustering to group semantically similar passages. Representative passages from each cluster are used to construct cluster-specific prompts for a large language model (LLM), generating intermediate answers that are filtered, reranked, and finally synthesized into a single, comprehensive response. This multi-stage pipeline enhances answer diversity, relevance, and faithfulness to retrieved evidence. Evaluated on the FineWeb Sample-10BT dataset, TopClustRAG ranked 2nd in faithfulness and 7th in correctness on the official leaderboard, demonstrating the effectiveness of clustering-based context filtering and prompt aggregation in large-scale RAG systems. 

---
# Modeling the One-to-Many Property in Open-Domain Dialogue with LLMs 

**Authors**: Jing Yang Lee, Kong-Aik Lee, Woon-Seng Gan  

**Link**: [PDF](https://arxiv.org/pdf/2506.15131)  

**Abstract**: Open-domain Dialogue (OD) exhibits a one-to-many (o2m) property, whereby multiple appropriate responses exist for a single dialogue context. Despite prior research showing that modeling this property boosts response diversity, most modern LLM-based dialogue agents do not explicitly do so. In this work, we model the o2m property of OD in LLMs by decomposing OD generation into two key tasks: Multi-Response Generation (MRG) and Preference-based Selection (PS), which entail generating a set of n semantically and lexically diverse high-quality responses for a given dialogue context, followed by selecting a single response based on human preference, respectively. To facilitate MRG and PS, we introduce o2mDial, a dialogue corpus explicitly designed to capture the o2m property by featuring multiple plausible responses for each context. Leveraging o2mDial, we propose new in-context learning and instruction-tuning strategies, as well as novel evaluation metrics for MRG, alongside a model-based approach for PS. Empirical results demonstrate that applying the proposed two-stage framework to smaller LLMs for OD generation enhances overall response diversity while maintaining contextual coherence, improving response quality by up to 90%, bringing them closer to the performance of larger models. 

---
# Learning-Time Encoding Shapes Unlearning in LLMs 

**Authors**: Ruihan Wu, Konstantin Garov, Kamalika Chaudhuri  

**Link**: [PDF](https://arxiv.org/pdf/2506.15076)  

**Abstract**: As large language models (LLMs) are increasingly deployed in the real world, the ability to ``unlearn'', or remove specific pieces of knowledge post hoc, has become essential for a variety of reasons ranging from privacy regulations to correcting outdated or harmful content. Prior work has proposed unlearning benchmarks and algorithms, and has typically assumed that the training process and the target model are fixed. In this work, we empirically investigate how learning-time choices in knowledge encoding impact the effectiveness of unlearning factual knowledge. Our experiments reveal two key findings: (1) learning with paraphrased descriptions improves unlearning performance and (2) unlearning individual piece of knowledge from a chunk of text is challenging. Our results suggest that learning-time knowledge encoding may play a central role in enabling reliable post-hoc unlearning. 

---
# Memory Tokens: Large Language Models Can Generate Reversible Sentence Embeddings 

**Authors**: Ignacio Sastre, Aiala Rosá  

**Link**: [PDF](https://arxiv.org/pdf/2506.15001)  

**Abstract**: In this work, we observe an interesting phenomenon: it is possible to generate reversible sentence embeddings that allow an LLM to reconstruct the original text exactly, without modifying the model's weights. This is achieved by introducing a special memory token, whose embedding is optimized through training on a fixed sequence. When prompted with this embedding, the model reconstructs the fixed sequence exactly. We evaluate this phenomenon across English and Spanish datasets, sequences of up to approximately 240 tokens, and model scales ranging from 100M to 8B parameters. Notably, Llama 3.1 8B successfully reconstructs all tested sequences. Our findings highlight an interesting capability of LLMs and suggest potential applications in memory-based retrieval, compression, and controlled text generation. 

---
# Lost in Variation? Evaluating NLI Performance in Basque and Spanish Geographical Variants 

**Authors**: Jaione Bengoetxea, Itziar Gonzalez-Dios, Rodrigo Agerri  

**Link**: [PDF](https://arxiv.org/pdf/2506.15239)  

**Abstract**: In this paper, we evaluate the capacity of current language technologies to understand Basque and Spanish language varieties. We use Natural Language Inference (NLI) as a pivot task and introduce a novel, manually-curated parallel dataset in Basque and Spanish, along with their respective variants. Our empirical analysis of crosslingual and in-context learning experiments using encoder-only and decoder-based Large Language Models (LLMs) shows a performance drop when handling linguistic variation, especially in Basque. Error analysis suggests that this decline is not due to lexical overlap, but rather to the linguistic variation itself. Further ablation experiments indicate that encoder-only models particularly struggle with Western Basque, which aligns with linguistic theory that identifies peripheral dialects (e.g., Western) as more distant from the standard. All data and code are publicly available. 

---
# From Chat to Checkup: Can Large Language Models Assist in Diabetes Prediction? 

**Authors**: Shadman Sakib, Oishy Fatema Akhand, Ajwad Abrar  

**Link**: [PDF](https://arxiv.org/pdf/2506.14949)  

**Abstract**: While Machine Learning (ML) and Deep Learning (DL) models have been widely used for diabetes prediction, the use of Large Language Models (LLMs) for structured numerical data is still not well explored. In this study, we test the effectiveness of LLMs in predicting diabetes using zero-shot, one-shot, and three-shot prompting methods. We conduct an empirical analysis using the Pima Indian Diabetes Database (PIDD). We evaluate six LLMs, including four open-source models: Gemma-2-27B, Mistral-7B, Llama-3.1-8B, and Llama-3.2-2B. We also test two proprietary models: GPT-4o and Gemini Flash 2.0. In addition, we compare their performance with three traditional machine learning models: Random Forest, Logistic Regression, and Support Vector Machine (SVM). We use accuracy, precision, recall, and F1-score as evaluation metrics. Our results show that proprietary LLMs perform better than open-source ones, with GPT-4o and Gemma-2-27B achieving the highest accuracy in few-shot settings. Notably, Gemma-2-27B also outperforms the traditional ML models in terms of F1-score. However, there are still issues such as performance variation across prompting strategies and the need for domain-specific fine-tuning. This study shows that LLMs can be useful for medical prediction tasks and encourages future work on prompt engineering and hybrid approaches to improve healthcare predictions. 

---
# MDBench: A Synthetic Multi-Document Reasoning Benchmark Generated with Knowledge Guidance 

**Authors**: Joseph J. Peper, Wenzhao Qiu, Ali Payani, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14927)  

**Abstract**: Natural language processing evaluation has made significant progress, largely driven by the proliferation of powerful large language mod-els (LLMs). New evaluation benchmarks are of increasing priority as the reasoning capabilities of LLMs are expanding at a rapid pace. In particular, while multi-document (MD) reasoning is an area of extreme relevance given LLM capabilities in handling longer-context inputs, few benchmarks exist to rigorously examine model behavior in this setting. Moreover, the multi-document setting is historically challenging for benchmark creation due to the expensive cost of annotating long inputs. In this work, we introduce MDBench, a new dataset for evaluating LLMs on the task of multi-document reasoning. Notably, MDBench is created through a novel synthetic generation process, allowing us to controllably and efficiently generate challenging document sets and the corresponding question-answer (QA) examples. Our novel technique operates on condensed structured seed knowledge, modifying it through LLM-assisted edits to induce MD-specific reasoning challenges. We then convert this structured knowledge into a natural text surface form, generating a document set and corresponding QA example. We analyze the behavior of popular LLMs and prompting techniques, finding that MDBENCH poses significant challenges for all methods, even with relatively short document sets. We also see our knowledge-guided generation technique (1) allows us to readily perform targeted analysis of MD-specific reasoning capabilities and (2) can be adapted quickly to account for new challenges and future modeling improvements. 

---
# CrEst: Credibility Estimation for Contexts in LLMs via Weak Supervision 

**Authors**: Dyah Adila, Shuai Zhang, Boran Han, Bonan Min, Yuyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14912)  

**Abstract**: The integration of contextual information has significantly enhanced the performance of large language models (LLMs) on knowledge-intensive tasks. However, existing methods often overlook a critical challenge: the credibility of context documents can vary widely, potentially leading to the propagation of unreliable information. In this paper, we introduce CrEst, a novel weakly supervised framework for assessing the credibility of context documents during LLM inference--without requiring manual annotations. Our approach is grounded in the insight that credible documents tend to exhibit higher semantic coherence with other credible documents, enabling automated credibility estimation through inter-document agreement. To incorporate credibility into LLM inference, we propose two integration strategies: a black-box approach for models without access to internal weights or activations, and a white-box method that directly modifies attention mechanisms. Extensive experiments across three model architectures and five datasets demonstrate that CrEst consistently outperforms strong baselines, achieving up to a 26.86% improvement in accuracy and a 3.49% increase in F1 score. Further analysis shows that CrEst maintains robust performance even under high-noise conditions. 

---
# ProtoReasoning: Prototypes as the Foundation for Generalizable Reasoning in LLMs 

**Authors**: Feng He, Zijun Chen, Xinnian Liang, Tingting Ma, Yunqi Qiu, Shuangzhi Wu, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.15211)  

**Abstract**: Recent advances in Large Reasoning Models (LRMs) trained with Long Chain-of-Thought (Long CoT) reasoning have demonstrated remarkable cross-domain generalization capabilities. However, the underlying mechanisms supporting such transfer remain poorly understood. We hypothesize that cross-domain generalization arises from shared abstract reasoning prototypes -- fundamental reasoning patterns that capture the essence of problems across domains. These prototypes minimize the nuances of the representation, revealing that seemingly diverse tasks are grounded in shared reasoning this http URL on this hypothesis, we propose ProtoReasoning, a framework that enhances the reasoning ability of LLMs by leveraging scalable and verifiable prototypical representations (Prolog for logical reasoning, PDDL for planning).ProtoReasoning features: (1) an automated prototype construction pipeline that transforms problems into corresponding prototype representations; (2) a comprehensive verification system providing reliable feedback through Prolog/PDDL interpreters; (3) the scalability to synthesize problems arbitrarily within prototype space while ensuring correctness. Extensive experiments show that ProtoReasoning achieves 4.7% improvement over baseline models on logical reasoning (Enigmata-Eval), 6.3% improvement on planning tasks, 4.0% improvement on general reasoning (MMLU) and 1.0% on mathematics (AIME24). Significantly, our ablation studies confirm that learning in prototype space also demonstrates enhanced generalization to structurally similar problems compared to training solely on natural language representations, validating our hypothesis that reasoning prototypes serve as the foundation for generalizable reasoning in large language models. 

---
# AutoRule: Reasoning Chain-of-thought Extracted Rule-based Rewards Improve Preference Learning 

**Authors**: Tevin Wang, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.15651)  

**Abstract**: Rule-based rewards offer a promising strategy for improving reinforcement learning from human feedback (RLHF), but current approaches often rely on manual rule engineering. We present AutoRule, a fully automated method for extracting rules from preference feedback and formulating them into rule-based rewards. AutoRule extraction operates in three stages: it leverages a reasoning model to interpret user preferences, identifies candidate rules from the reasoning chain of these interpretations, and synthesizes them into a unified rule set. Leveraging the finalized rule set, we employ language-model verifiers to compute the fraction of rules satisfied by each output, using this metric as an auxiliary reward alongside the learned reward model during policy optimization. Training a Llama-3-8B model with AutoRule results in a 28.6\% relative improvement in length-controlled win rate on AlpacaEval2.0, and a 6.1\% relative gain in second-turn performance on a held-out MT-Bench subset, compared to a GRPO baseline trained with the same learned reward model but without the rule-based auxiliary reward. Our analysis confirms that the extracted rules exhibit good agreement with dataset preference. We find that AutoRule demonstrates reduced reward hacking compared to a learned reward model when run over two episodes. Finally, our case study suggests that the extracted rules capture unique qualities valued in different datasets. The extracted rules are provided in the appendix, and the code is open-sourced at this https URL. 

---
# video-SALMONN 2: Captioning-Enhanced Audio-Visual Large Language Models 

**Authors**: Changli Tang, Yixuan Li, Yudong Yang, Jimin Zhuang, Guangzhi Sun, Wei Li, Zejun Ma, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15220)  

**Abstract**: Videos contain a wealth of information, and generating detailed and accurate descriptions in natural language is a key aspect of video understanding. In this paper, we present video-SALMONN 2, an advanced audio-visual large language model (LLM) with low-rank adaptation (LoRA) designed for enhanced video (with paired audio) captioning through directed preference optimisation (DPO). We propose new metrics to evaluate the completeness and accuracy of video descriptions, which are optimised using DPO. To further improve training, we propose a novel multi-round DPO (MrDPO) approach, which involves periodically updating the DPO reference model, merging and re-initialising the LoRA module as a proxy for parameter updates after each training round (1,000 steps), and incorporating guidance from ground-truth video captions to stabilise the process. Experimental results show that MrDPO significantly enhances video-SALMONN 2's captioning accuracy, reducing the captioning error rates by 28\%. The final video-SALMONN 2 model, with just 7 billion parameters, surpasses leading models such as GPT-4o and Gemini-1.5-Pro in video captioning tasks, while maintaining highly competitive performance to the state-of-the-art on widely used video question-answering benchmarks among models of similar size. Codes are available at \href{this https URL}{this https URL}. 

---
# Hypothesis Testing for Quantifying LLM-Human Misalignment in Multiple Choice Settings 

**Authors**: Harbin Hong, Sebastian Caldas, Liu Leqi  

**Link**: [PDF](https://arxiv.org/pdf/2506.14997)  

**Abstract**: As Large Language Models (LLMs) increasingly appear in social science research (e.g., economics and marketing), it becomes crucial to assess how well these models replicate human behavior. In this work, using hypothesis testing, we present a quantitative framework to assess the misalignment between LLM-simulated and actual human behaviors in multiple-choice survey settings. This framework allows us to determine in a principled way whether a specific language model can effectively simulate human opinions, decision-making, and general behaviors represented through multiple-choice options. We applied this framework to a popular language model for simulating people's opinions in various public surveys and found that this model is ill-suited for simulating the tested sub-populations (e.g., across different races, ages, and incomes) for contentious questions. This raises questions about the alignment of this language model with the tested populations, highlighting the need for new practices in using LLMs for social science studies beyond naive simulations of human subjects. 

---
# Cost-Efficient Serving of LLM Agents via Test-Time Plan Caching 

**Authors**: Qizheng Zhang, Michael Wornow, Kunle Olukotun  

**Link**: [PDF](https://arxiv.org/pdf/2506.14852)  

**Abstract**: LLM-based agentic applications have shown increasingly remarkable capabilities in complex workflows but incur substantial costs due to extensive planning and reasoning requirements. Existing LLM caching techniques (like context caching and semantic caching), primarily designed for serving chatbots, are insufficient for agentic applications where outputs depend on external data or environmental contexts. We propose agentic plan caching, a novel approach that extracts, stores, adapts, and reuses structured plan templates from planning stages of agentic applications across semantically similar tasks to reduce the cost of serving. Unlike traditional semantic caching, our system extracts plan templates from completed agent executions at test-time, employs keyword extraction to match new requests against cached plans, and utilizes lightweight models to adapt these templates to task-specific plans with contexts. Evaluation across multiple real-world agentic applications shows that our system can reduce costs by 46.62% on average while maintaining performance, offering a more efficient solution for serving LLM-based agents that complements existing LLM serving infrastructures. 

---
# Argus Inspection: Do Multimodal Large Language Models Possess the Eye of Panoptes? 

**Authors**: Yang Yao, Lingyu Li, Jiaxin Song, Chiyu Chen, Zhenqi He, Yixu Wang, Xin Wang, Tianle Gu, Jie Li, Yan Teng, Yingchun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14805)  

**Abstract**: As Multimodal Large Language Models (MLLMs) continue to evolve, their cognitive and reasoning capabilities have seen remarkable progress. However, challenges in visual fine-grained perception and commonsense causal inference persist. This paper introduces Argus Inspection, a multimodal benchmark with two levels of difficulty, emphasizing detailed visual recognition while incorporating real-world commonsense understanding to evaluate causal reasoning abilities. Expanding on it, we present the Eye of Panoptes framework, which integrates a binary parametric Sigmoid metric with an indicator function, enabling a more holistic evaluation of MLLMs' responses in opinion-based reasoning tasks. Experiments conducted on 26 mainstream MLLMs reveal that the highest performance in visual fine-grained reasoning reaches only 0.46, highlighting considerable potential for enhancement. Our research offers valuable perspectives for the continued refinement of MLLMs. 

---
# Identifying economic narratives in large text corpora -- An integrated approach using Large Language Models 

**Authors**: Tobias Schmidt, Kai-Robin Lange, Matthias Reccius, Henrik Müller, Michael Roos, Carsten Jentsch  

**Link**: [PDF](https://arxiv.org/pdf/2506.15041)  

**Abstract**: As interest in economic narratives has grown in recent years, so has the number of pipelines dedicated to extracting such narratives from texts. Pipelines often employ a mix of state-of-the-art natural language processing techniques, such as BERT, to tackle this task. While effective on foundational linguistic operations essential for narrative extraction, such models lack the deeper semantic understanding required to distinguish extracting economic narratives from merely conducting classic tasks like Semantic Role Labeling. Instead of relying on complex model pipelines, we evaluate the benefits of Large Language Models (LLMs) by analyzing a corpus of Wall Street Journal and New York Times newspaper articles about inflation. We apply a rigorous narrative definition and compare GPT-4o outputs to gold-standard narratives produced by expert annotators. Our results suggests that GPT-4o is capable of extracting valid economic narratives in a structured format, but still falls short of expert-level performance when handling complex documents and narratives. Given the novelty of LLMs in economic research, we also provide guidance for future work in economics and the social sciences that employs LLMs to pursue similar objectives. 

---
# Revisiting Reinforcement Learning for LLM Reasoning from A Cross-Domain Perspective 

**Authors**: Zhoujun Cheng, Shibo Hao, Tianyang Liu, Fan Zhou, Yutao Xie, Feng Yao, Yuexin Bian, Yonghao Zhuang, Nilabjo Dey, Yuheng Zha, Yi Gu, Kun Zhou, Yuqi Wang, Yuan Li, Richard Fan, Jianshu She, Chengqian Gao, Abulhair Saparov, Haonan Li, Taylor W. Killian, Mikhail Yurochkin, Zhengzhong Liu, Eric P. Xing, Zhiting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14965)  

**Abstract**: Reinforcement learning (RL) has emerged as a promising approach to improve large language model (LLM) reasoning, yet most open efforts focus narrowly on math and code, limiting our understanding of its broader applicability to general reasoning. A key challenge lies in the lack of reliable, scalable RL reward signals across diverse reasoning domains. We introduce Guru, a curated RL reasoning corpus of 92K verifiable examples spanning six reasoning domains--Math, Code, Science, Logic, Simulation, and Tabular--each built through domain-specific reward design, deduplication, and filtering to ensure reliability and effectiveness for RL training. Based on Guru, we systematically revisit established findings in RL for LLM reasoning and observe significant variation across domains. For example, while prior work suggests that RL primarily elicits existing knowledge from pretrained models, our results reveal a more nuanced pattern: domains frequently seen during pretraining (Math, Code, Science) easily benefit from cross-domain RL training, while domains with limited pretraining exposure (Logic, Simulation, and Tabular) require in-domain training to achieve meaningful performance gains, suggesting that RL is likely to facilitate genuine skill acquisition. Finally, we present Guru-7B and Guru-32B, two models that achieve state-of-the-art performance among open models RL-trained with publicly available data, outperforming best baselines by 7.9% and 6.7% on our 17-task evaluation suite across six reasoning domains. We also show that our models effectively improve the Pass@k performance of their base models, particularly on complex tasks less likely to appear in pretraining data. We release data, models, training and evaluation code to facilitate general-purpose reasoning at: this https URL 

---
# RE-IMAGINE: Symbolic Benchmark Synthesis for Reasoning Evaluation 

**Authors**: Xinnuo Xu, Rachel Lawrence, Kshitij Dubey, Atharva Pandey, Risa Ueno, Fabian Falck, Aditya V. Nori, Rahul Sharma, Amit Sharma, Javier Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2506.15455)  

**Abstract**: Recent Large Language Models (LLMs) have reported high accuracy on reasoning benchmarks. However, it is still unclear whether the observed results arise from true reasoning or from statistical recall of the training set. Inspired by the ladder of causation (Pearl, 2009) and its three levels (associations, interventions and counterfactuals), this paper introduces RE-IMAGINE, a framework to characterize a hierarchy of reasoning ability in LLMs, alongside an automated pipeline to generate problem variations at different levels of the hierarchy. By altering problems in an intermediate symbolic representation, RE-IMAGINE generates arbitrarily many problems that are not solvable using memorization alone. Moreover, the framework is general and can work across reasoning domains, including math, code, and logic. We demonstrate our framework on four widely-used benchmarks to evaluate several families of LLMs, and observe reductions in performance when the models are queried with problem variations. These assessments indicate a degree of reliance on statistical recall for past performance, and open the door to further research targeting skills across the reasoning hierarchy. 

---
# Exploring and Exploiting the Inherent Efficiency within Large Reasoning Models for Self-Guided Efficiency Enhancement 

**Authors**: Weixiang Zhao, Jiahe Guo, Yang Deng, Xingyu Sui, Yulin Hu, Yanyan Zhao, Wanxiang Che, Bing Qin, Tat-Seng Chua, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15647)  

**Abstract**: Recent advancements in large reasoning models (LRMs) have significantly enhanced language models' capabilities in complex problem-solving by emulating human-like deliberative thinking. However, these models often exhibit overthinking (i.e., the generation of unnecessarily verbose and redundant content), which hinders efficiency and inflates inference cost. In this work, we explore the representational and behavioral origins of this inefficiency, revealing that LRMs inherently possess the capacity for more concise reasoning. Empirical analyses show that correct reasoning paths vary significantly in length, and the shortest correct responses often suffice, indicating untapped efficiency potential. Exploiting these findings, we propose two lightweight methods to enhance LRM efficiency. First, we introduce Efficiency Steering, a training-free activation steering technique that modulates reasoning behavior via a single direction in the model's representation space. Second, we develop Self-Rewarded Efficiency RL, a reinforcement learning framework that dynamically balances task accuracy and brevity by rewarding concise correct solutions. Extensive experiments on seven LRM backbones across multiple mathematical reasoning benchmarks demonstrate that our methods significantly reduce reasoning length while preserving or improving task performance. Our results highlight that reasoning efficiency can be improved by leveraging and guiding the intrinsic capabilities of existing models in a self-guided manner. 

---
# Managing Complex Failure Analysis Workflows with LLM-based Reasoning and Acting Agents 

**Authors**: Aline Dobrovsky, Konstantin Schekotihin, Christian Burmer  

**Link**: [PDF](https://arxiv.org/pdf/2506.15567)  

**Abstract**: Failure Analysis (FA) is a highly intricate and knowledge-intensive process. The integration of AI components within the computational infrastructure of FA labs has the potential to automate a variety of tasks, including the detection of non-conformities in images, the retrieval of analogous cases from diverse data sources, and the generation of reports from annotated images. However, as the number of deployed AI models increases, the challenge lies in orchestrating these components into cohesive and efficient workflows that seamlessly integrate with the FA process.
This paper investigates the design and implementation of a Large Language Model (LLM)-based Planning Agent (LPA) to assist FA engineers in solving their analysis cases. The LPA integrates LLMs with advanced planning capabilities and external tool utilization, enabling autonomous processing of complex queries, retrieval of relevant data from external systems, and generation of human-readable responses. Evaluation results demonstrate the agent's operational effectiveness and reliability in supporting FA tasks. 

---
# HeurAgenix: Leveraging LLMs for Solving Complex Combinatorial Optimization Challenges 

**Authors**: Xianliang Yang, Ling Zhang, Haolong Qian, Lei Song, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2506.15196)  

**Abstract**: Heuristic algorithms play a vital role in solving combinatorial optimization (CO) problems, yet traditional designs depend heavily on manual expertise and struggle to generalize across diverse instances. We introduce \textbf{HeurAgenix}, a two-stage hyper-heuristic framework powered by large language models (LLMs) that first evolves heuristics and then selects among them automatically. In the heuristic evolution phase, HeurAgenix leverages an LLM to compare seed heuristic solutions with higher-quality solutions and extract reusable evolution strategies. During problem solving, it dynamically picks the most promising heuristic for each problem state, guided by the LLM's perception ability. For flexibility, this selector can be either a state-of-the-art LLM or a fine-tuned lightweight model with lower inference cost. To mitigate the scarcity of reliable supervision caused by CO complexity, we fine-tune the lightweight heuristic selector with a dual-reward mechanism that jointly exploits singals from selection preferences and state perception, enabling robust selection under noisy annotations. Extensive experiments on canonical benchmarks show that HeurAgenix not only outperforms existing LLM-based hyper-heuristics but also matches or exceeds specialized solvers. Code is available at this https URL. 

---
# The Effect of State Representation on LLM Agent Behavior in Dynamic Routing Games 

**Authors**: Lyle Goodyear, Rachel Guo, Ramesh Johari  

**Link**: [PDF](https://arxiv.org/pdf/2506.15624)  

**Abstract**: Large Language Models (LLMs) have shown promise as decision-makers in dynamic settings, but their stateless nature necessitates creating a natural language representation of history. We present a unifying framework for systematically constructing natural language "state" representations for prompting LLM agents in repeated multi-agent games. Previous work on games with LLM agents has taken an ad hoc approach to encoding game history, which not only obscures the impact of state representation on agents' behavior, but also limits comparability between studies. Our framework addresses these gaps by characterizing methods of state representation along three axes: action informativeness (i.e., the extent to which the state representation captures actions played); reward informativeness (i.e., the extent to which the state representation describes rewards obtained); and prompting style (or natural language compression, i.e., the extent to which the full text history is summarized).
We apply this framework to a dynamic selfish routing game, chosen because it admits a simple equilibrium both in theory and in human subject experiments \cite{rapoport_choice_2009}. Despite the game's relative simplicity, we find that there are key dependencies of LLM agent behavior on the natural language state representation. In particular, we observe that representations which provide agents with (1) summarized, rather than complete, natural language representations of past history; (2) information about regrets, rather than raw payoffs; and (3) limited information about others' actions lead to behavior that more closely matches game theoretic equilibrium predictions, and with more stable game play by the agents. By contrast, other representations can exhibit either large deviations from equilibrium, higher variation in dynamic game play over time, or both. 

---
# RePCS: Diagnosing Data Memorization in LLM-Powered Retrieval-Augmented Generation 

**Authors**: Le Vu Anh, Nguyen Viet Anh, Mehmet Dik, Luong Van Nghia  

**Link**: [PDF](https://arxiv.org/pdf/2506.15513)  

**Abstract**: Retrieval-augmented generation (RAG) has become a common strategy for updating large language model (LLM) responses with current, external information. However, models may still rely on memorized training data, bypass the retrieved evidence, and produce contaminated outputs. We introduce Retrieval-Path Contamination Scoring (RePCS), a diagnostic method that detects such behavior without requiring model access or retraining. RePCS compares two inference paths: (i) a parametric path using only the query, and (ii) a retrieval-augmented path using both the query and retrieved context by computing the Kullback-Leibler (KL) divergence between their output distributions. A low divergence suggests that the retrieved context had minimal impact, indicating potential memorization. This procedure is model-agnostic, requires no gradient or internal state access, and adds only a single additional forward pass. We further derive PAC-style guarantees that link the KL threshold to user-defined false positive and false negative rates. On the Prompt-WNQA benchmark, RePCS achieves a ROC-AUC of 0.918. This result outperforms the strongest prior method by 6.5 percentage points while keeping latency overhead below 4.7% on an NVIDIA T4 GPU. RePCS offers a lightweight, black-box safeguard to verify whether a RAG system meaningfully leverages retrieval, making it especially valuable in safety-critical applications. 

---
# Uncovering Intention through LLM-Driven Code Snippet Description Generation 

**Authors**: Yusuf Sulistyo Nugroho, Farah Danisha Salam, Brittany Reid, Raula Gaikovina Kula, Kazumasa Shimari, Kenichi Matsumoto  

**Link**: [PDF](https://arxiv.org/pdf/2506.15453)  

**Abstract**: Documenting code snippets is essential to pinpoint key areas where both developers and users should pay attention. Examples include usage examples and other Application Programming Interfaces (APIs), which are especially important for third-party libraries. With the rise of Large Language Models (LLMs), the key goal is to investigate the kinds of description developers commonly use and evaluate how well an LLM, in this case Llama, can support description generation. We use NPM Code Snippets, consisting of 185,412 packages with 1,024,579 code snippets. From there, we use 400 code snippets (and their descriptions) as samples. First, our manual classification found that the majority of original descriptions (55.5%) highlight example-based usage. This finding emphasizes the importance of clear documentation, as some descriptions lacked sufficient detail to convey intent. Second, the LLM correctly identified the majority of original descriptions as "Example" (79.75%), which is identical to our manual finding, showing a propensity for generalization. Third, compared to the originals, the produced description had an average similarity score of 0.7173, suggesting relevance but room for improvement. Scores below 0.9 indicate some irrelevance. Our results show that depending on the task of the code snippet, the intention of the document may differ from being instructions for usage, installations, or descriptive learning examples for any user of a library. 

---
# Unlocking Post-hoc Dataset Inference with Synthetic Data 

**Authors**: Bihe Zhao, Pratyush Maini, Franziska Boenisch, Adam Dziedzic  

**Link**: [PDF](https://arxiv.org/pdf/2506.15271)  

**Abstract**: The remarkable capabilities of Large Language Models (LLMs) can be mainly attributed to their massive training datasets, which are often scraped from the internet without respecting data owners' intellectual property rights. Dataset Inference (DI) offers a potential remedy by identifying whether a suspect dataset was used in training, thereby enabling data owners to verify unauthorized use. However, existing DI methods require a private set-known to be absent from training-that closely matches the compromised dataset's distribution. Such in-distribution, held-out data is rarely available in practice, severely limiting the applicability of DI. In this work, we address this challenge by synthetically generating the required held-out set. Our approach tackles two key obstacles: (1) creating high-quality, diverse synthetic data that accurately reflects the original distribution, which we achieve via a data generator trained on a carefully designed suffix-based completion task, and (2) bridging likelihood gaps between real and synthetic data, which is realized through post-hoc calibration. Extensive experiments on diverse text datasets show that using our generated data as a held-out set enables DI to detect the original training sets with high confidence, while maintaining a low false positive rate. This result empowers copyright owners to make legitimate claims on data usage and demonstrates our method's reliability for real-world litigations. Our code is available at this https URL. 

---
# LLM Agent for Hyper-Parameter Optimization 

**Authors**: Wanzhe Wang, Jianqiu Peng, Menghao Hu, Weihuang Zhong, Tong Zhang, Shuai Wang, Yixin Zhang, Mingjie Shao, Wanli Ni  

**Link**: [PDF](https://arxiv.org/pdf/2506.15167)  

**Abstract**: Hyper-parameters are essential and critical for the performance of communication algorithms. However, current hyper-parameters tuning methods for warm-start particles swarm optimization with cross and mutation (WS-PSO-CM) algortihm for radio map-enabled unmanned aerial vehicle (UAV) trajectory and communication are primarily heuristic-based, exhibiting low levels of automation and unsatisfactory performance. In this paper, we design an large language model (LLM) agent for automatic hyper-parameters-tuning, where an iterative framework and model context protocol (MCP) are applied. In particular, the LLM agent is first setup via a profile, which specifies the mission, background, and output format. Then, the LLM agent is driven by the prompt requirement, and iteratively invokes WS-PSO-CM algorithm for exploration. Finally, the LLM agent autonomously terminates the loop and returns a set of hyper-parameters. Our experiment results show that the minimal sum-rate achieved by hyper-parameters generated via our LLM agent is significantly higher than those by both human heuristics and random generation methods. This indicates that an LLM agent with PSO knowledge and WS-PSO-CM algorithm background is useful in finding high-performance hyper-parameters. 

---
# Singular Value Decomposition on Kronecker Adaptation for Large Language Model 

**Authors**: Yee Hin Chong, Peng Qu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15251)  

**Abstract**: Large pre-trained Transformer models achieve state-of-the-art results across diverse language and reasoning tasks, but full fine-tuning incurs substantial storage, memory, and computational overhead. Parameter-efficient fine-tuning (PEFT) methods mitigate these costs by learning only a small subset of task-specific parameters, yet existing approaches either introduce inference-time latency (adapter modules), suffer from suboptimal convergence (randomly initialized low-rank updates), or rely on fixed rank choices that may not match task complexity (Kronecker-based decompositions).
We propose SoKA (SVD on Kronecker Adaptation), a novel PEFT strategy that combines Kronecker-product tensor factorization with SVD-driven initialization and spectrum-aware dynamic rank selection. Our Kronecker-Product SVD (KPSVD) procedure extracts principal components of the full weight update into compact Kronecker factors, while an adaptive rank selection algorithm uses energy-threshold and elbow-point criteria to prune negligible components.
Empirical evaluation on LLaMA2-7B across arithmetic reasoning (GSM8K), formal mathematics (MATH), and code generation (MBPP) demonstrates that SoKA requires only 0.99M trainable parameters, 25% fewer than LoRA/PiSSA, while matching or exceeding baseline performance. Moreover, SoKA exhibits faster convergence and more stable gradients, highlighting its robustness and efficiency for large-scale model adaptation. 

---
# SFT-GO: Supervised Fine-Tuning with Group Optimization for Large Language Models 

**Authors**: Gyuhak Kim, Sumiran Singh Thakur, Su Min Park, Wei Wei, Yujia Bao  

**Link**: [PDF](https://arxiv.org/pdf/2506.15021)  

**Abstract**: Supervised fine-tuning (SFT) has become an essential step in tailoring large language models (LLMs) to align with human expectations and specific downstream tasks. However, existing SFT methods typically treat each training instance as a uniform sequence, giving equal importance to all tokens regardless of their relevance. This overlooks the fact that only a subset of tokens often contains critical, task-specific information. To address this limitation, we introduce Supervised Fine-Tuning with Group Optimization (SFT-GO), a novel approach that treats groups of tokens differently based on their this http URL-GO groups tokens in each sample based on their importance values and optimizes the LLM using a weighted combination of the worst-group loss and the standard cross-entropy loss. This mechanism adaptively emphasizes the most challenging token groups and guides the model to better handle different group distributions, thereby improving overall learning dynamics. We provide a theoretical analysis of SFT-GO's convergence rate, demonstrating its efficiency. Empirically, we apply SFT-GO with three different token grouping strategies and show that models trained with SFT-GO consistently outperform baseline approaches across popular LLM benchmarks. These improvements hold across various datasets and base models, demonstrating the robustness and the effectiveness of our method. 

---
# Thinking in Directivity: Speech Large Language Model for Multi-Talker Directional Speech Recognition 

**Authors**: Jiamin Xie, Ju Lin, Yiteng Huang, Tyler Vuong, Zhaojiang Lin, Zhaojun Yang, Peng Su, Prashant Rawat, Sangeeta Srivastava, Ming Sun, Florian Metze  

**Link**: [PDF](https://arxiv.org/pdf/2506.14973)  

**Abstract**: Recent studies have demonstrated that prompting large language models (LLM) with audio encodings enables effective speech recognition capabilities. However, the ability of Speech LLMs to comprehend and process multi-channel audio with spatial cues remains a relatively uninvestigated area of research. In this work, we present directional-SpeechLlama, a novel approach that leverages the microphone array of smart glasses to achieve directional speech recognition, source localization, and bystander cross-talk suppression. To enhance the model's ability to understand directivity, we propose two key techniques: serialized directional output training (S-DOT) and contrastive direction data augmentation (CDDA). Experimental results show that our proposed directional-SpeechLlama effectively captures the relationship between textual cues and spatial audio, yielding strong performance in both speech recognition and source localization tasks. 

---
# Mapping Caregiver Needs to AI Chatbot Design: Strengths and Gaps in Mental Health Support for Alzheimer's and Dementia Caregivers 

**Authors**: Jiayue Melissa Shi, Dong Whi Yoo, Keran Wang, Violeta J. Rodriguez, Ravi Karkar, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2506.15047)  

**Abstract**: Family caregivers of individuals with Alzheimer's Disease and Related Dementia (AD/ADRD) face significant emotional and logistical challenges that place them at heightened risk for stress, anxiety, and depression. Although recent advances in generative AI -- particularly large language models (LLMs) -- offer new opportunities to support mental health, little is known about how caregivers perceive and engage with such technologies. To address this gap, we developed Carey, a GPT-4o-based chatbot designed to provide informational and emotional support to AD/ADRD caregivers. Using Carey as a technology probe, we conducted semi-structured interviews with 16 family caregivers following scenario-driven interactions grounded in common caregiving stressors. Through inductive coding and reflexive thematic analysis, we surface a systemic understanding of caregiver needs and expectations across six themes -- on-demand information access, emotional support, safe space for disclosure, crisis management, personalization, and data privacy. For each of these themes, we also identified the nuanced tensions in the caregivers' desires and concerns. We present a mapping of caregiver needs, AI chatbot's strengths, gaps, and design recommendations. Our findings offer theoretical and practical insights to inform the design of proactive, trustworthy, and caregiver-centered AI systems that better support the evolving mental health needs of AD/ADRD caregivers. 

---
# FedNano: Toward Lightweight Federated Tuning for Pretrained Multimodal Large Language Models 

**Authors**: Yao Zhang, Hewei Gao, Haokun Chen, Weiguo Li, Yunpu Ma, Volker Tresp  

**Link**: [PDF](https://arxiv.org/pdf/2506.14824)  

**Abstract**: Multimodal Large Language Models (MLLMs) excel in tasks like multimodal reasoning and cross-modal retrieval but face deployment challenges in real-world scenarios due to distributed multimodal data and strict privacy requirements. Federated Learning (FL) offers a solution by enabling collaborative model training without centralizing data. However, realizing FL for MLLMs presents significant challenges, including high computational demands, limited client capacity, substantial communication costs, and heterogeneous client data. Existing FL methods assume client-side deployment of full models, an assumption that breaks down for large-scale MLLMs due to their massive size and communication demands. To address these limitations, we propose FedNano, the first FL framework that centralizes the LLM on the server while introducing NanoEdge, a lightweight module for client-specific adaptation. NanoEdge employs modality-specific encoders, connectors, and trainable NanoAdapters with low-rank adaptation. This design eliminates the need to deploy LLM on clients, reducing client-side storage by 95%, and limiting communication overhead to only 0.01% of the model parameters. By transmitting only compact NanoAdapter updates, FedNano handles heterogeneous client data and resource constraints while preserving privacy. Experiments demonstrate that FedNano outperforms prior FL baselines, bridging the gap between MLLM scale and FL feasibility, and enabling scalable, decentralized multimodal AI systems. 

---
# MedSyn: Enhancing Diagnostics with Human-AI Collaboration 

**Authors**: Burcu Sayin, Ipek Baris Schlicht, Ngoc Vo Hong, Sara Allievi, Jacopo Staiano, Pasquale Minervini, Andrea Passerini  

**Link**: [PDF](https://arxiv.org/pdf/2506.14774)  

**Abstract**: Clinical decision-making is inherently complex, often influenced by cognitive biases, incomplete information, and case ambiguity. Large Language Models (LLMs) have shown promise as tools for supporting clinical decision-making, yet their typical one-shot or limited-interaction usage may overlook the complexities of real-world medical practice. In this work, we propose a hybrid human-AI framework, MedSyn, where physicians and LLMs engage in multi-step, interactive dialogues to refine diagnoses and treatment decisions. Unlike static decision-support tools, MedSyn enables dynamic exchanges, allowing physicians to challenge LLM suggestions while the LLM highlights alternative perspectives. Through simulated physician-LLM interactions, we assess the potential of open-source LLMs as physician assistants. Results show open-source LLMs are promising as physician assistants in the real world. Future work will involve real physician interactions to further validate MedSyn's usefulness in diagnostic accuracy and patient outcomes. 

---
