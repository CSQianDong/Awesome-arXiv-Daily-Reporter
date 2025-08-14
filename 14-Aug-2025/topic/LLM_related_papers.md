# MEML-GRPO: Heterogeneous Multi-Expert Mutual Learning for RLVR Advancement 

**Authors**: Weitao Jia, Jinghui Lu, Haiyang Yu, Siqi Wang, Guozhi Tang, An-Lan Wang, Weijie Yin, Dingkang Yang, Yuxiang Nie, Bin Shan, Hao Feng, Irene Li, Kun Yang, Han Wang, Jingqun Tang, Teng Fu, Changhong Jin, Chao Feng, Xiaohui Lv, Can Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09670)  

**Abstract**: Recent advances demonstrate that reinforcement learning with verifiable rewards (RLVR) significantly enhances the reasoning capabilities of large language models (LLMs). However, standard RLVR faces challenges with reward sparsity, where zero rewards from consistently incorrect candidate answers provide no learning signal, particularly in challenging tasks. To address this, we propose Multi-Expert Mutual Learning GRPO (MEML-GRPO), an innovative framework that utilizes diverse expert prompts as system prompts to generate a broader range of responses, substantially increasing the likelihood of identifying correct solutions. Additionally, we introduce an inter-expert mutual learning mechanism that facilitates knowledge sharing and transfer among experts, further boosting the model's performance through RLVR. Extensive experiments across multiple reasoning benchmarks show that MEML-GRPO delivers significant improvements, achieving an average performance gain of 4.89% with Qwen and 11.33% with Llama, effectively overcoming the core limitations of traditional RLVR methods. 

---
# Mathematical Computation and Reasoning Errors by Large Language Models 

**Authors**: Liang Zhang, Edith Aurora Graf  

**Link**: [PDF](https://arxiv.org/pdf/2508.09932)  

**Abstract**: Large Language Models (LLMs) are increasingly utilized in AI-driven educational instruction and assessment, particularly within mathematics education. The capability of LLMs to generate accurate answers and detailed solutions for math problem-solving tasks is foundational for ensuring reliable and precise feedback and assessment in math education practices. Our study focuses on evaluating the accuracy of four LLMs (OpenAI GPT-4o and o1, DeepSeek-V3 and DeepSeek-R1) solving three categories of math tasks, including arithmetic, algebra, and number theory, and identifies step-level reasoning errors within their solutions. Instead of relying on standard benchmarks, we intentionally build math tasks (via item models) that are challenging for LLMs and prone to errors. The accuracy of final answers and the presence of errors in individual solution steps were systematically analyzed and coded. Both single-agent and dual-agent configurations were tested. It is observed that the reasoning-enhanced OpenAI o1 model consistently achieved higher or nearly perfect accuracy across all three math task categories. Analysis of errors revealed that procedural slips were the most frequent and significantly impacted overall performance, while conceptual misunderstandings were less frequent. Deploying dual-agent configurations substantially improved overall performance. These findings offer actionable insights into enhancing LLM performance and underscore effective strategies for integrating LLMs into mathematics education, thereby advancing AI-driven instructional practices and assessment precision. 

---
# EvoCurr: Self-evolving Curriculum with Behavior Code Generation for Complex Decision-making 

**Authors**: Yang Cheng, Zilai Wang, Weiyu Ma, Wenhui Zhu, Yue Deng, Jian Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.09586)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse domains, including programming, planning, and decision-making. However, their performance often degrades when faced with highly complex problem instances that require deep reasoning over long horizons. In such cases, direct problem-solving approaches can lead to inefficiency or failure due to the lack of structured intermediate guidance. To address this, we propose a novel self-evolve framework, EvoCurr, in which a dedicated curriculum-generation LLM constructs a sequence of problem instances with gradually increasing difficulty, tailored to the solver LLM's learning progress. The curriculum dynamically adapts easing challenges when the solver struggles and escalating them when success is consistent, thus maintaining an optimal learning trajectory. This approach enables the solver LLM, implemented as a code-generation model producing Python decision-tree scripts, to progressively acquire the skills needed for complex decision-making tasks. Experimental results on challenging decision-making benchmarks show that our method significantly improves task success rates and solution efficiency compared to direct-solving baselines. These findings suggest that LLM-driven curriculum learning holds strong potential for enhancing automated reasoning in real-world, high-complexity domains. 

---
# UDA: Unsupervised Debiasing Alignment for Pair-wise LLM-as-a-Judge 

**Authors**: Yang Zhang, Cunxiang Wang, Lindong Wu, Wenbo Yu, Yidong Wang, Guangsheng Bao, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09724)  

**Abstract**: Pairwise evaluation of Large Language Models (LLMs) is a common paradigm, but it is prone to preference bias, where judges systematically favor certain outputs, such as their own. This bias leads to inconsistent and skewed rankings across different judges. To address this, we first empirically demonstrate significant and heterogeneous biases in cross-model evaluations. We then propose UDA (Unsupervised Debiasing Alignment), a framework that reduces inter-judge disagreement by dynamically adjusting the Elo rating system. For each pairwise comparison, a compact neural network learns to adaptively set the K-factor and refine win probabilities. Crucially, UDA operates in a fully unsupervised manner, guided solely by the objective of minimizing the dispersion among the Elo trajectories of all judges. This forces an alignment towards a collective consensus, which serves as an unsupervised proxy for a more stable and reproducible evaluation. In addition, we provide theoretical motivation demonstrating how alignment towards a consensus can reduce aggregate system bias. Experiments show that UDA significantly reduces the inter-judge rating standard deviation by up to 63.4% and improves the average correlation with human judgments by 24.7%. Notably, UDA elevates the performance of poorly performing judges to achieve parity with high-quality ones, fostering a more robust and reliable evaluation ecosystem. Code and data are available at this https URL. 

---
# Beyond Naïve Prompting: Strategies for Improved Zero-shot Context-aided Forecasting with LLMs 

**Authors**: Arjun Ashok, Andrew Robert Williams, Vincent Zhihao Zheng, Irina Rish, Nicolas Chapados, Étienne Marcotte, Valentina Zantedeschi, Alexandre Drouin  

**Link**: [PDF](https://arxiv.org/pdf/2508.09904)  

**Abstract**: Forecasting in real-world settings requires models to integrate not only historical data but also relevant contextual information, often available in textual form. While recent work has shown that large language models (LLMs) can be effective context-aided forecasters via naïve direct prompting, their full potential remains underexplored. We address this gap with 4 strategies, providing new insights into the zero-shot capabilities of LLMs in this setting. ReDP improves interpretability by eliciting explicit reasoning traces, allowing us to assess the model's reasoning over the context independently from its forecast accuracy. CorDP leverages LLMs solely to refine existing forecasts with context, enhancing their applicability in real-world forecasting pipelines. IC-DP proposes embedding historical examples of context-aided forecasting tasks in the prompt, substantially improving accuracy even for the largest models. Finally, RouteDP optimizes resource efficiency by using LLMs to estimate task difficulty, and routing the most challenging tasks to larger models. Evaluated on different kinds of context-aided forecasting tasks from the CiK benchmark, our strategies demonstrate distinct benefits over naïve prompting across LLMs of different sizes and families. These results open the door to further simple yet effective improvements in LLM-based context-aided forecasting. 

---
# Memory Decoder: A Pretrained, Plug-and-Play Memory for Large Language Models 

**Authors**: Jiaqi Cao, Jiarui Wang, Rubin Wei, Qipeng Guo, Kai Chen, Bowen Zhou, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.09874)  

**Abstract**: Large Language Models (LLMs) have shown strong abilities in general language tasks, yet adapting them to specific domains remains a challenge. Current method like Domain Adaptive Pretraining (DAPT) requires costly full-parameter training and suffers from catastrophic forgetting. Meanwhile, Retrieval-Augmented Generation (RAG) introduces substantial inference latency due to expensive nearest-neighbor searches and longer context. This paper introduces Memory Decoder, a plug-and-play pretrained memory that enables efficient domain adaptation without changing the original model's parameters. Memory Decoder employs a small transformer decoder that learns to imitate the behavior of an external non-parametric retriever. Once trained, Memory Decoder can be seamlessly integrated with any pretrained language model that shares the same tokenizer, requiring no model-specific modifications. Experimental results demonstrate that Memory Decoder enables effective adaptation of various Qwen and Llama models to three distinct specialized domains: biomedicine, finance, and law, reducing perplexity by an average of 6.17 points. Overall, Memory Decoder introduces a novel paradigm centered on a specially pretrained memory component designed for domain-specific adaptation. This memory architecture can be integrated in a plug-and-play manner, consistently enhancing performance across multiple models within the target domain. 

---
# A Comprehensive Evaluation framework of Alignment Techniques for LLMs 

**Authors**: Muneeza Azmat, Momin Abbas, Maysa Malfiza Garcia de Macedo, Marcelo Carpinette Grave, Luan Soares de Souza, Tiago Machado, Rogerio A de Paula, Raya Horesh, Yixin Chen, Heloisa Caroline de Souza Pereira Candello, Rebecka Nordenlow, Aminat Adebiyi  

**Link**: [PDF](https://arxiv.org/pdf/2508.09937)  

**Abstract**: As Large Language Models (LLMs) become increasingly integrated into real-world applications, ensuring their outputs align with human values and safety standards has become critical. The field has developed diverse alignment approaches including traditional fine-tuning methods (RLHF, instruction tuning), post-hoc correction systems, and inference-time interventions, each with distinct advantages and limitations. However, the lack of unified evaluation frameworks makes it difficult to systematically compare these paradigms and guide deployment decisions. This paper introduces a multi-dimensional evaluation of alignment techniques for LLMs, a comprehensive evaluation framework that provides a systematic comparison across all major alignment paradigms. Our framework assesses methods along four key dimensions: alignment detection, alignment quality, computational efficiency, and robustness. Through experiments across diverse base models and alignment strategies, we demonstrate the utility of our framework in identifying strengths and limitations of current state-of-the-art models, providing valuable insights for future research directions. 

---
# Exploring the Potential of Large Language Models in Fine-Grained Review Comment Classification 

**Authors**: Linh Nguyen, Chunhua Liu, Hong Yi Lin, Patanamon Thongtanunam  

**Link**: [PDF](https://arxiv.org/pdf/2508.09832)  

**Abstract**: Code review is a crucial practice in software development. As code review nowadays is lightweight, various issues can be identified, and sometimes, they can be trivial. Research has investigated automated approaches to classify review comments to gauge the effectiveness of code reviews. However, previous studies have primarily relied on supervised machine learning, which requires extensive manual annotation to train the models effectively. To address this limitation, we explore the potential of using Large Language Models (LLMs) to classify code review comments. We assess the performance of LLMs to classify 17 categories of code review comments. Our results show that LLMs can classify code review comments, outperforming the state-of-the-art approach using a trained deep learning model. In particular, LLMs achieve better accuracy in classifying the five most useful categories, which the state-of-the-art approach struggles with due to low training examples. Rather than relying solely on a specific small training data distribution, our results show that LLMs provide balanced performance across high- and low-frequency categories. These results suggest that the LLMs could offer a scalable solution for code review analytics to improve the effectiveness of the code review process. 

---
# LibRec: Benchmarking Retrieval-Augmented LLMs for Library Migration Recommendations 

**Authors**: Junxiao Han, Yarong Wang, Xiaodong Gu, Cuiyun Gao, Yao Wan, Song Han, David Lo, Shuiguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2508.09791)  

**Abstract**: In this paper, we propose LibRec, a novel framework that integrates the capabilities of LLMs with retrieval-augmented generation(RAG) techniques to automate the recommendation of alternative libraries. The framework further employs in-context learning to extract migration intents from commit messages to enhance the accuracy of its recommendations. To evaluate the effectiveness of LibRec, we introduce LibEval, a benchmark designed to assess the performance in the library migration recommendation task. LibEval comprises 2,888 migration records associated with 2,368 libraries extracted from 2,324 Python repositories. Each migration record captures source-target library pairs, along with their corresponding migration intents and intent types. Based on LibEval, we evaluated the effectiveness of ten popular LLMs within our framework, conducted an ablation study to examine the contributions of key components within our framework, explored the impact of various prompt strategies on the framework's performance, assessed its effectiveness across various intent types, and performed detailed failure case analyses. 

---
# Can LLM-Generated Textual Explanations Enhance Model Classification Performance? An Empirical Study 

**Authors**: Mahdi Dhaini, Juraj Vladika, Ege Erdogan, Zineb Attaoui, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2508.09776)  

**Abstract**: In the rapidly evolving field of Explainable Natural Language Processing (NLP), textual explanations, i.e., human-like rationales, are pivotal for explaining model predictions and enriching datasets with interpretable labels. Traditional approaches rely on human annotation, which is costly, labor-intensive, and impedes scalability. In this work, we present an automated framework that leverages multiple state-of-the-art large language models (LLMs) to generate high-quality textual explanations. We rigorously assess the quality of these LLM-generated explanations using a comprehensive suite of Natural Language Generation (NLG) metrics. Furthermore, we investigate the downstream impact of these explanations on the performance of pre-trained language models (PLMs) and LLMs across natural language inference tasks on two diverse benchmark datasets. Our experiments demonstrate that automated explanations exhibit highly competitive effectiveness compared to human-annotated explanations in improving model performance. Our findings underscore a promising avenue for scalable, automated LLM-based textual explanation generation for extending NLP datasets and enhancing model performance. 

---
# Evaluating the Role of Large Language Models in Legal Practice in India 

**Authors**: Rahul Hemrajani  

**Link**: [PDF](https://arxiv.org/pdf/2508.09713)  

**Abstract**: The integration of Artificial Intelligence(AI) into the legal profession raises significant questions about the capacity of Large Language Models(LLM) to perform key legal tasks. In this paper, I empirically evaluate how well LLMs, such as GPT, Claude, and Llama, perform key legal tasks in the Indian context, including issue spotting, legal drafting, advice, research, and reasoning. Through a survey experiment, I compare outputs from LLMs with those of a junior lawyer, with advanced law students rating the work on helpfulness, accuracy, and comprehensiveness. LLMs excel in drafting and issue spotting, often matching or surpassing human work. However, they struggle with specialised legal research, frequently generating hallucinations, factually incorrect or fabricated outputs. I conclude that while LLMs can augment certain legal tasks, human expertise remains essential for nuanced reasoning and the precise application of law. 

---
# AmbiGraph-Eval: Can LLMs Effectively Handle Ambiguous Graph Queries? 

**Authors**: Yuchen Tian, Kaixin Li, Hao Chen, Ziyang Luo, Hongzhan Lin, Sebastian Schelter, Lun Du, Jing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.09631)  

**Abstract**: Large Language Models (LLMs) have recently demonstrated strong capabilities in translating natural language into database queries, especially when dealing with complex graph-structured data. However, real-world queries often contain inherent ambiguities, and the interconnected nature of graph structures can amplify these challenges, leading to unintended or incorrect query results. To systematically evaluate LLMs on this front, we propose a taxonomy of graph-query ambiguities, comprising three primary types: Attribute Ambiguity, Relationship Ambiguity, and Attribute-Relationship Ambiguity, each subdivided into Same-Entity and Cross-Entity scenarios. We introduce AmbiGraph-Eval, a novel benchmark of real-world ambiguous queries paired with expert-verified graph query answers. Evaluating 9 representative LLMs shows that even top models struggle with ambiguous graph queries. Our findings reveal a critical gap in ambiguity handling and motivate future work on specialized resolution techniques. 

---
# Improving ARDS Diagnosis Through Context-Aware Concept Bottleneck Models 

**Authors**: Anish Narain, Ritam Majumdar, Nikita Narayanan, Dominic Marshall, Sonali Parbhoo  

**Link**: [PDF](https://arxiv.org/pdf/2508.09719)  

**Abstract**: Large, publicly available clinical datasets have emerged as a novel resource for understanding disease heterogeneity and to explore personalization of therapy. These datasets are derived from data not originally collected for research purposes and, as a result, are often incomplete and lack critical labels. Many AI tools have been developed to retrospectively label these datasets, such as by performing disease classification; however, they often suffer from limited interpretability. Previous work has attempted to explain predictions using Concept Bottleneck Models (CBMs), which learn interpretable concepts that map to higher-level clinical ideas, facilitating human evaluation. However, these models often experience performance limitations when the concepts fail to adequately explain or characterize the task. We use the identification of Acute Respiratory Distress Syndrome (ARDS) as a challenging test case to demonstrate the value of incorporating contextual information from clinical notes to improve CBM performance. Our approach leverages a Large Language Model (LLM) to process clinical notes and generate additional concepts, resulting in a 10% performance gain over existing methods. Additionally, it facilitates the learning of more comprehensive concepts, thereby reducing the risk of information leakage and reliance on spurious shortcuts, thus improving the characterization of ARDS. 

---
# Your Coding Intent is Secretly in the Context and You Should Deliberately Infer It Before Completion 

**Authors**: Yanzhou Li, Tianlin Li, Yiran Zhang, Shangqing Liu, Aishan Liu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.09537)  

**Abstract**: Large Language Models (LLMs) are increasingly used for function completion in repository-scale codebases. Prior studies demonstrate that when explicit instructions--such as docstrings--are provided, these models can generate highly accurate implementations. However, in real-world repositories, such annotations are frequently absent, and performance drops substantially without them. To address this gap, we frame the task as a three-stage process. The first stage focuses on intent inference, where the model analyzes the code preceding the target function to uncover cues about the desired functionality. Such preceding context often encodes subtle but critical information, and we design a reasoning-based prompting framework to guide the LLM through step-by-step extraction and synthesis of these signals before any code is generated. The second stage introduces an optional interactive refinement mechanism to handle cases where preceding context alone is insufficient for intent recovery. In this stage, the model proposes a small set of candidate intentions, enabling the developer to select or edit them so that the inferred intent closely matches the actual requirement. Finally, in the third stage, the LLM generates the target function conditioned on the finalized intent. To support this pipeline, we curate a dataset of 40,000 examples annotated with intermediate reasoning traces and corresponding docstrings. Extensive experiments on DevEval and ComplexCodeEval show that our approach consistently boosts multiple LLMs, achieving over 20\% relative gains in both reference-based and execution-based metrics, with the interactive refinement stage delivering additional improvements beyond these gains. 

---
# Interpretable Robot Control via Structured Behavior Trees and Large Language Models 

**Authors**: Ingrid Maéva Chekam, Ines Pastor-Martinez, Ali Tourani, Jose Andres Millan-Romera, Laura Ribeiro, Pedro Miguel Bastos Soares, Holger Voos, Jose Luis Sanchez-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2508.09621)  

**Abstract**: As intelligent robots become more integrated into human environments, there is a growing need for intuitive and reliable Human-Robot Interaction (HRI) interfaces that are adaptable and more natural to interact with. Traditional robot control methods often require users to adapt to interfaces or memorize predefined commands, limiting usability in dynamic, unstructured environments. This paper presents a novel framework that bridges natural language understanding and robotic execution by combining Large Language Models (LLMs) with Behavior Trees. This integration enables robots to interpret natural language instructions given by users and translate them into executable actions by activating domain-specific plugins. The system supports scalable and modular integration, with a primary focus on perception-based functionalities, such as person tracking and hand gesture recognition. To evaluate the system, a series of real-world experiments was conducted across diverse environments. Experimental results demonstrate that the proposed approach is practical in real-world scenarios, with an average cognition-to-execution accuracy of approximately 94%, making a significant contribution to HRI systems and robots. The complete source code of the framework is publicly available at this https URL. 

---
# TimeMKG: Knowledge-Infused Causal Reasoning for Multivariate Time Series Modeling 

**Authors**: Yifei Sun, Junming Liu, Ding Wang, Yirong Chen, Xuefeng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2508.09630)  

**Abstract**: Multivariate time series data typically comprises two distinct modalities: variable semantics and sampled numerical observations. Traditional time series models treat variables as anonymous statistical signals, overlooking the rich semantic information embedded in variable names and data descriptions. However, these textual descriptors often encode critical domain knowledge that is essential for robust and interpretable modeling. Here we present TimeMKG, a multimodal causal reasoning framework that elevates time series modeling from low-level signal processing to knowledge informed inference. TimeMKG employs large language models to interpret variable semantics and constructs structured Multivariate Knowledge Graphs that capture inter-variable relationships. A dual-modality encoder separately models the semantic prompts, generated from knowledge graph triplets, and the statistical patterns from historical time series. Cross-modality attention aligns and fuses these representations at the variable level, injecting causal priors into downstream tasks such as forecasting and classification, providing explicit and interpretable priors to guide model reasoning. The experiment in diverse datasets demonstrates that incorporating variable-level knowledge significantly improves both predictive performance and generalization. 

---
# DeepFeatIoT: Unifying Deep Learned, Randomized, and LLM Features for Enhanced IoT Time Series Sensor Data Classification in Smart Industries 

**Authors**: Muhammad Sakib Khan Inan, Kewen Liao  

**Link**: [PDF](https://arxiv.org/pdf/2508.09468)  

**Abstract**: Internet of Things (IoT) sensors are ubiquitous technologies deployed across smart cities, industrial sites, and healthcare systems. They continuously generate time series data that enable advanced analytics and automation in industries. However, challenges such as the loss or ambiguity of sensor metadata, heterogeneity in data sources, varying sampling frequencies, inconsistent units of measurement, and irregular timestamps make raw IoT time series data difficult to interpret, undermining the effectiveness of smart systems. To address these challenges, we propose a novel deep learning model, DeepFeatIoT, which integrates learned local and global features with non-learned randomized convolutional kernel-based features and features from large language models (LLMs). This straightforward yet unique fusion of diverse learned and non-learned features significantly enhances IoT time series sensor data classification, even in scenarios with limited labeled data. Our model's effectiveness is demonstrated through its consistent and generalized performance across multiple real-world IoT sensor datasets from diverse critical application domains, outperforming state-of-the-art benchmark models. These results highlight DeepFeatIoT's potential to drive significant advancements in IoT analytics and support the development of next-generation smart systems. 

---
# GoViG: Goal-Conditioned Visual Navigation Instruction Generation 

**Authors**: Fengyi Wu, Yifei Dong, Zhi-Qi Cheng, Yilong Dai, Guangyu Chen, Hang Wang, Qi Dai, Alexander G. Hauptmann  

**Link**: [PDF](https://arxiv.org/pdf/2508.09547)  

**Abstract**: We introduce Goal-Conditioned Visual Navigation Instruction Generation (GoViG), a new task that aims to autonomously generate precise and contextually coherent navigation instructions solely from egocentric visual observations of initial and goal states. Unlike conventional approaches that rely on structured inputs such as semantic annotations or environmental maps, GoViG exclusively leverages raw egocentric visual data, substantially improving its adaptability to unseen and unstructured environments. Our method addresses this task by decomposing it into two interconnected subtasks: (1) visual forecasting, which predicts intermediate visual states bridging the initial and goal views; and (2) instruction generation, which synthesizes linguistically coherent instructions grounded in both observed and anticipated visuals. These subtasks are integrated within an autoregressive multimodal large language model trained with tailored objectives to ensure spatial accuracy and linguistic clarity. Furthermore, we introduce two complementary multimodal reasoning strategies, one-pass and interleaved reasoning, to mimic incremental human cognitive processes during navigation. To evaluate our method, we propose the R2R-Goal dataset, combining diverse synthetic and real-world trajectories. Empirical results demonstrate significant improvements over state-of-the-art methods, achieving superior BLEU-4 and CIDEr scores along with robust cross-domain generalization. 

---
# Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference 

**Authors**: Zhifan Luo, Shuo Shao, Su Zhang, Lijing Zhou, Yuke Hu, Chenxu Zhao, Zhihao Liu, Zhan Qin  

**Link**: [PDF](https://arxiv.org/pdf/2508.09442)  

**Abstract**: The Key-Value (KV) cache, which stores intermediate attention computations (Key and Value pairs) to avoid redundant calculations, is a fundamental mechanism for accelerating Large Language Model (LLM) inference. However, this efficiency optimization introduces significant yet underexplored privacy risks. This paper provides the first comprehensive analysis of these vulnerabilities, demonstrating that an attacker can reconstruct sensitive user inputs directly from the KV-cache. We design and implement three distinct attack vectors: a direct Inversion Attack, a more broadly applicable and potent Collision Attack, and a semantic-based Injection Attack. These methods demonstrate the practicality and severity of KV-cache privacy leakage issues. To mitigate this, we propose KV-Cloak, a novel, lightweight, and efficient defense mechanism. KV-Cloak uses a reversible matrix-based obfuscation scheme, combined with operator fusion, to secure the KV-cache. Our extensive experiments show that KV-Cloak effectively thwarts all proposed attacks, reducing reconstruction quality to random noise. Crucially, it achieves this robust security with virtually no degradation in model accuracy and minimal performance overhead, offering a practical solution for trustworthy LLM deployment. 

---
# NeuronTune: Fine-Grained Neuron Modulation for Balanced Safety-Utility Alignment in LLMs 

**Authors**: Birong Pan, Mayi Xu, Qiankun Pi, Jianhao Chen, Yuanyuan Zhu, Ming Zhong, Tieyun Qian  

**Link**: [PDF](https://arxiv.org/pdf/2508.09473)  

**Abstract**: Ensuring robust safety alignment while preserving utility is critical for the reliable deployment of Large Language Models (LLMs). However, current techniques fundamentally suffer from intertwined deficiencies: insufficient robustness against malicious attacks, frequent refusal of benign queries, degradation in generated text quality and general task performance--the former two reflecting deficits in robust safety and the latter constituting utility impairment. We trace these limitations to the coarse-grained layer-wise interventions in existing methods. To resolve this, we propose NeuronTune, a fine-grained framework that dynamically modulates sparse neurons to achieve simultaneous safety-utility optimization. Our approach first identifies safety-critical and utility-preserving neurons across all layers via attribution, then employs meta-learning to adaptively amplify safety-neuron activations and suppress utility-neuron activations. Crucially, NeuronTune enables tunable adjustment of intervention scope via neuron-count thresholds, supporting flexible adaptation to security-critical or utility-priority scenarios. Extensive experimental results demonstrate that our method significantly outperforms existing state-of-the-art technologies, achieving superior model safety while maintaining excellent utility. 

---
# Hallucination vs interpretation: rethinking accuracy and precision in AI-assisted data extraction for knowledge synthesis 

**Authors**: Xi Long, Christy Boscardin, Lauren A. Maggio, Joseph A. Costello, Ralph Gonzales, Rasmyah Hammoudeh, Ki Lai, Yoon Soo Park, Brian C. Gin  

**Link**: [PDF](https://arxiv.org/pdf/2508.09458)  

**Abstract**: Knowledge syntheses (literature reviews) are essential to health professions education (HPE), consolidating findings to advance theory and practice. However, they are labor-intensive, especially during data extraction. Artificial Intelligence (AI)-assisted extraction promises efficiency but raises concerns about accuracy, making it critical to distinguish AI 'hallucinations' (fabricated content) from legitimate interpretive differences. We developed an extraction platform using large language models (LLMs) to automate data extraction and compared AI to human responses across 187 publications and 17 extraction questions from a published scoping review. AI-human, human-human, and AI-AI consistencies were measured using interrater reliability (categorical) and thematic similarity ratings (open-ended). Errors were identified by comparing extracted responses to source publications. AI was highly consistent with humans for concrete, explicitly stated questions (e.g., title, aims) and lower for questions requiring subjective interpretation or absent in text (e.g., Kirkpatrick's outcomes, study rationale). Human-human consistency was not higher than AI-human and showed the same question-dependent variability. Discordant AI-human responses (769/3179 = 24.2%) were mostly due to interpretive differences (18.3%); AI inaccuracies were rare (1.51%), while humans were nearly three times more likely to state inaccuracies (4.37%). Findings suggest AI accuracy depends more on interpretability than hallucination. Repeating AI extraction can identify interpretive complexity or ambiguity, refining processes before human review. AI can be a transparent, trustworthy partner in knowledge synthesis, though caution is needed to preserve critical human insights. 

---
# APIO: Automatic Prompt Induction and Optimization for Grammatical Error Correction and Text Simplification 

**Authors**: Artem Chernodub, Aman Saini, Yejin Huh, Vivek Kulkarni, Vipul Raheja  

**Link**: [PDF](https://arxiv.org/pdf/2508.09378)  

**Abstract**: Recent advancements in large language models (LLMs) have enabled a wide range of natural language processing (NLP) tasks to be performed through simple prompt-based interactions. Consequently, several approaches have been proposed to engineer prompts that most effectively enable LLMs to perform a given task (e.g., chain-of-thought prompting). In settings with a well-defined metric to optimize model performance, automatic prompt optimization (APO) methods have been developed to refine a seed prompt. Advancing this line of research, we propose APIO, a simple but effective prompt induction and optimization approach for the tasks of Grammatical Error Correction (GEC) and Text Simplification, without relying on manually specified seed prompts. APIO achieves a new state-of-the-art performance for purely LLM-based prompting methods on these tasks. We make our data, code, prompts, and outputs publicly available. 

---
# Leveraging Large Language Models for Rare Disease Named Entity Recognition 

**Authors**: Nan Miles Xi, Yu Deng, Lin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09323)  

**Abstract**: Named Entity Recognition (NER) in the rare disease domain poses unique challenges due to limited labeled data, semantic ambiguity between entity types, and long-tail distributions. In this study, we evaluate the capabilities of GPT-4o for rare disease NER under low-resource settings, using a range of prompt-based strategies including zero-shot prompting, few-shot in-context learning, retrieval-augmented generation (RAG), and task-level fine-tuning. We design a structured prompting framework that encodes domain-specific knowledge and disambiguation rules for four entity types. We further introduce two semantically guided few-shot example selection methods to improve in-context performance while reducing labeling effort. Experiments on the RareDis Corpus show that GPT-4o achieves competitive or superior performance compared to BioClinicalBERT, with task-level fine-tuning yielding new state-of-the-art (SOTA) results. Cost-performance analysis reveals that few-shot prompting delivers high returns at low token budgets, while RAG offers marginal additional benefit. An error taxonomy highlights common failure modes such as boundary drift and type confusion, suggesting opportunities for post-processing and hybrid refinement. Our results demonstrate that prompt-optimized LLMs can serve as effective, scalable alternatives to traditional supervised models in biomedical NER, particularly in rare disease applications where annotated data is scarce. 

---
# ParallelSearch: Train your LLMs to Decompose Query and Search Sub-queries in Parallel with Reinforcement Learning 

**Authors**: Shu Zhao, Tan Yu, Anbang Xu, Japinder Singh, Aaditya Shukla, Rama Akkiraju  

**Link**: [PDF](https://arxiv.org/pdf/2508.09303)  

**Abstract**: Reasoning-augmented search agents such as Search-R1, trained via reinforcement learning with verifiable rewards (RLVR), demonstrate remarkable capabilities in multi-step information retrieval from external knowledge sources. These agents address the limitations of their parametric memory by dynamically gathering relevant facts to address complex reasoning tasks. However, existing approaches suffer from a fundamental architectural limitation: they process search queries strictly sequentially, even when handling inherently parallelizable and logically independent comparisons. This sequential bottleneck significantly constrains computational efficiency, particularly for queries that require multiple entity comparisons. To address this critical limitation, we propose ParallelSearch, a novel reinforcement learning framework that empowers large language models (LLMs) to recognize parallelizable query structures and execute multiple search operations concurrently. Our approach introduces dedicated reward functions that incentivize the identification of independent query components while preserving answer accuracy through jointly considering correctness, query decomposition quality, and parallel execution benefits. Comprehensive experiments demonstrate that ParallelSearch outperforms state-of-the-art baselines by an average performance gain of 2.9% across seven question-answering benchmarks. Notably, on parallelizable questions, our method achieves a 12.7% performance improvement while requiring only 69.6% of the LLM calls compared to sequential approaches. 

---
# NEFMind: Parameter-Efficient Fine-Tuning of Open-Source LLMs for Telecom APIs Automation 

**Authors**: Zainab Khan, Ahmed Hussain, Mukesh Thakur, Arto Hellas, Panos Papadimitratos  

**Link**: [PDF](https://arxiv.org/pdf/2508.09240)  

**Abstract**: The use of Service-Based Architecture in modern telecommunications has exponentially increased Network Functions (NFs) and Application Programming Interfaces (APIs), creating substantial operational complexities in service discovery and management. We introduce \textit{NEFMind}, a framework leveraging parameter-efficient fine-tuning of open-source Large Language Models (LLMs) to address these challenges. It integrates three core components: synthetic dataset generation from Network Exposure Function (NEF) API specifications, model optimization through Quantized-Low-Rank Adaptation, and performance evaluation via GPT-4 Ref Score and BertScore metrics. Targeting 5G Service-Based Architecture APIs, our approach achieves 85% reduction in communication overhead compared to manual discovery methods. Experimental validation using the open-source Phi-2 model demonstrates exceptional API call identification performance at 98-100% accuracy. The fine-tuned Phi-2 model delivers performance comparable to significantly larger models like GPT-4 while maintaining computational efficiency for telecommunications infrastructure deployment. These findings validate domain-specific, parameter-efficient LLM strategies for managing complex API ecosystems in next-generation telecommunications networks. 

---
# AMRG: Extend Vision Language Models for Automatic Mammography Report Generation 

**Authors**: Nak-Jun Sung, Donghyun Lee, Bo Hwa Choi, Chae Jung Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.09225)  

**Abstract**: Mammography report generation is a critical yet underexplored task in medical AI, characterized by challenges such as multiview image reasoning, high-resolution visual cues, and unstructured radiologic language. In this work, we introduce AMRG (Automatic Mammography Report Generation), the first end-to-end framework for generating narrative mammography reports using large vision-language models (VLMs). Building upon MedGemma-4B-it-a domain-specialized, instruction-tuned VLM-we employ a parameter-efficient fine-tuning (PEFT) strategy via Low-Rank Adaptation (LoRA), enabling lightweight adaptation with minimal computational overhead. We train and evaluate AMRG on DMID, a publicly available dataset of paired high-resolution mammograms and diagnostic reports. This work establishes the first reproducible benchmark for mammography report generation, addressing a longstanding gap in multimodal clinical AI. We systematically explore LoRA hyperparameter configurations and conduct comparative experiments across multiple VLM backbones, including both domain-specific and general-purpose models under a unified tuning protocol. Our framework demonstrates strong performance across both language generation and clinical metrics, achieving a ROUGE-L score of 0.5691, METEOR of 0.6152, CIDEr of 0.5818, and BI-RADS accuracy of 0.5582. Qualitative analysis further highlights improved diagnostic consistency and reduced hallucinations. AMRG offers a scalable and adaptable foundation for radiology report generation and paves the way for future research in multimodal medical AI. 

---
# AI Blob! LLM-Driven Recontextualization of Italian Television Archives 

**Authors**: Roberto Balestri  

**Link**: [PDF](https://arxiv.org/pdf/2508.09535)  

**Abstract**: This paper introduces AI Blob!, an experimental system designed to explore the potential of semantic cataloging and Large Language Models (LLMs) for the retrieval and recontextualization of archival television footage. Drawing methodological inspiration from Italian television programs such as Blob (RAI Tre, 1989-), AI Blob! integrates automatic speech recognition (ASR), semantic embeddings, and retrieval-augmented generation (RAG) to organize and reinterpret archival content. The system processes a curated dataset of 1,547 Italian television videos by transcribing audio, segmenting it into sentence-level units, and embedding these segments into a vector database for semantic querying. Upon user input of a thematic prompt, the LLM generates a range of linguistically and conceptually related queries, guiding the retrieval and recombination of audiovisual fragments. These fragments are algorithmically selected and structured into narrative sequences producing montages that emulate editorial practices of ironic juxtaposition and thematic coherence. By foregrounding dynamic, content-aware retrieval over static metadata schemas, AI Blob! demonstrates how semantic technologies can facilitate new approaches to archival engagement, enabling novel forms of automated narrative construction and cultural analysis. The project contributes to ongoing debates in media historiography and AI-driven archival research, offering both a conceptual framework and a publicly available dataset to support further interdisciplinary experimentation. 

---
# MME-Emotion: A Holistic Evaluation Benchmark for Emotional Intelligence in Multimodal Large Language Models 

**Authors**: Fan Zhang, Zebang Cheng, Chong Deng, Haoxuan Li, Zheng Lian, Qian Chen, Huadai Liu, Wen Wang, Yi-Fan Zhang, Renrui Zhang, Ziyu Guo, Zhihong Zhu, Hao Wu, Haixin Wang, Yefeng Zheng, Xiaojiang Peng, Xian Wu, Kun Wang, Xiangang Li, Jieping Ye, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2508.09210)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have catalyzed transformative progress in affective computing, enabling models to exhibit emergent emotional intelligence. Despite substantial methodological progress, current emotional benchmarks remain limited, as it is still unknown: (a) the generalization abilities of MLLMs across distinct scenarios, and (b) their reasoning capabilities to identify the triggering factors behind emotional states. To bridge these gaps, we present \textbf{MME-Emotion}, a systematic benchmark that assesses both emotional understanding and reasoning capabilities of MLLMs, enjoying \textit{scalable capacity}, \textit{diverse settings}, and \textit{unified protocols}. As the largest emotional intelligence benchmark for MLLMs, MME-Emotion contains over 6,000 curated video clips with task-specific questioning-answering (QA) pairs, spanning broad scenarios to formulate eight emotional tasks. It further incorporates a holistic evaluation suite with hybrid metrics for emotion recognition and reasoning, analyzed through a multi-agent system framework. Through a rigorous evaluation of 20 advanced MLLMs, we uncover both their strengths and limitations, yielding several key insights: \ding{182} Current MLLMs exhibit unsatisfactory emotional intelligence, with the best-performing model achieving only $39.3\%$ recognition score and $56.0\%$ Chain-of-Thought (CoT) score on our benchmark. \ding{183} Generalist models (\emph{e.g.}, Gemini-2.5-Pro) derive emotional intelligence from generalized multimodal understanding capabilities, while specialist models (\emph{e.g.}, R1-Omni) can achieve comparable performance through domain-specific post-training adaptation. By introducing MME-Emotion, we hope that it can serve as a foundation for advancing MLLMs' emotional intelligence in the future. 

---
# MX-AI: Agentic Observability and Control Platform for Open and AI-RAN 

**Authors**: Ilias Chatzistefanidis, Andrea Leone, Ali Yaghoubian, Mikel Irazabal, Sehad Nassim, Lina Bariah, Merouane Debbah, Navid Nikaein  

**Link**: [PDF](https://arxiv.org/pdf/2508.09197)  

**Abstract**: Future 6G radio access networks (RANs) will be artificial intelligence (AI)-native: observed, reasoned about, and re-configured by autonomous agents cooperating across the cloud-edge continuum. We introduce MX-AI, the first end-to-end agentic system that (i) instruments a live 5G Open RAN testbed based on OpenAirInterface (OAI) and FlexRIC, (ii) deploys a graph of Large-Language-Model (LLM)-powered agents inside the Service Management and Orchestration (SMO) layer, and (iii) exposes both observability and control functions for 6G RAN resources through natural-language intents. On 50 realistic operational queries, MX-AI attains a mean answer quality of 4.1/5.0 and 100 % decision-action accuracy, while incurring only 8.8 seconds end-to-end latency when backed by GPT-4.1. Thus, it matches human-expert performance, validating its practicality in real settings. We publicly release the agent graph, prompts, and evaluation harness to accelerate open research on AI-native RANs. A live demo is presented here: this https URL 

---
# Towards Effective MLLM Jailbreaking Through Balanced On-Topicness and OOD-Intensity 

**Authors**: Zuoou Li, Weitong Zhang, Jingyuan Wang, Shuyuan Zhang, Wenjia Bai, Bernhard Kainz, Mengyun Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.09218)  

**Abstract**: Multimodal large language models (MLLMs) are widely used in vision-language reasoning tasks. However, their vulnerability to adversarial prompts remains a serious concern, as safety mechanisms often fail to prevent the generation of harmful outputs. Although recent jailbreak strategies report high success rates, many responses classified as "successful" are actually benign, vague, or unrelated to the intended malicious goal. This mismatch suggests that current evaluation standards may overestimate the effectiveness of such attacks. To address this issue, we introduce a four-axis evaluation framework that considers input on-topicness, input out-of-distribution (OOD) intensity, output harmfulness, and output refusal rate. This framework identifies truly effective jailbreaks. In a substantial empirical study, we reveal a structural trade-off: highly on-topic prompts are frequently blocked by safety filters, whereas those that are too OOD often evade detection but fail to produce harmful content. However, prompts that balance relevance and novelty are more likely to evade filters and trigger dangerous output. Building on this insight, we develop a recursive rewriting strategy called Balanced Structural Decomposition (BSD). The approach restructures malicious prompts into semantically aligned sub-tasks, while introducing subtle OOD signals and visual cues that make the inputs harder to detect. BSD was tested across 13 commercial and open-source MLLMs, where it consistently led to higher attack success rates, more harmful outputs, and fewer refusals. Compared to previous methods, it improves success rates by $67\%$ and harmfulness by $21\%$, revealing a previously underappreciated weakness in current multimodal safety systems. 

---
# From Values to Tokens: An LLM-Driven Framework for Context-aware Time Series Forecasting via Symbolic Discretization 

**Authors**: Xiaoyu Tao, Shilong Zhang, Mingyue Cheng, Daoyu Wang, Tingyue Pan, Bokai Pan, Changqing Zhang, Shijin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09191)  

**Abstract**: Time series forecasting plays a vital role in supporting decision-making across a wide range of critical applications, including energy, healthcare, and finance. Despite recent advances, forecasting accuracy remains limited due to the challenge of integrating historical numerical sequences with contextual features, which often comprise unstructured textual data. To address this challenge, we propose TokenCast, an LLM-driven framework that leverages language-based symbolic representations as a unified intermediary for context-aware time series forecasting. Specifically, TokenCast employs a discrete tokenizer to transform continuous numerical sequences into temporal tokens, enabling structural alignment with language-based inputs. To bridge the semantic gap between modalities, both temporal and contextual tokens are embedded into a shared representation space via a pre-trained large language model (LLM), further optimized with autoregressive generative objectives. Building upon this unified semantic space, the aligned LLM is subsequently fine-tuned in a supervised manner to predict future temporal tokens, which are then decoded back into the original numerical space. Extensive experiments on diverse real-world datasets enriched with contextual features demonstrate the effectiveness and generalizability of TokenCast. 

---
# Fine-Grained Safety Neurons with Training-Free Continual Projection to Reduce LLM Fine Tuning Risks 

**Authors**: Bing Han, Feifei Zhao, Dongcheng Zhao, Guobin Shen, Ping Wu, Yu Shi, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2508.09190)  

**Abstract**: Fine-tuning as service injects domain-specific knowledge into large language models (LLMs), while challenging the original alignment mechanisms and introducing safety risks. A series of defense strategies have been proposed for the alignment, fine-tuning, and post-fine-tuning phases, where most post-fine-tuning defenses rely on coarse-grained safety layer mapping. These methods lack a comprehensive consideration of both safety layers and fine-grained neurons, limiting their ability to efficiently balance safety and utility. To address this, we propose the Fine-Grained Safety Neurons (FGSN) with Training-Free Continual Projection method to reduce the fine-tuning safety risks. FGSN inherently integrates the multi-scale interactions between safety layers and neurons, localizing sparser and more precise fine-grained safety neurons while minimizing interference with downstream task neurons. We then project the safety neuron parameters onto safety directions, improving model safety while aligning more closely with human preferences. Extensive experiments across multiple fine-tuned LLM models demonstrate that our method significantly reduce harmfulness scores and attack success rates with minimal parameter modifications, while preserving the model's utility. Furthermore, by introducing a task-specific, multi-dimensional heterogeneous safety neuron cluster optimization mechanism, we achieve continual defense and generalization capability against unforeseen emerging safety concerns. 

---
# Motif 2.6B Technical Report 

**Authors**: Junghwan Lim, Sungmin Lee, Dongseok Kim, Eunhwan Park, Hyunbyung Park, Junhyeok Lee, Wai Ting Cheung, Dahye Choi, Jaeheui Her, Jaeyeon Huh, Hanbin Jung, Changjin Kang, Beomgyu Kim, Jihwan Kim, Minjae Kim, Taehwan Kim, Youngrok Kim, Haesol Lee, Jeesoo Lee, Kungyu Lee, Dongpin Oh, Yeongjae Park, Bokki Ryu, Daewon Suh, Dongjoo Weon  

**Link**: [PDF](https://arxiv.org/pdf/2508.09148)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have revolutionized artificial intelligence, yet developing an effective foundational LLM that balances high performance with computational efficiency remains challenging, especially for emerging research groups. To address this gap, we introduce Motif-2.6B, a 2.6-billion-parameter foundation model designed to democratize advanced LLM capabilities. Motif-2.6B incorporates several innovative architectural enhancements, including Differential Attention and PolyNorm activation functions, which improve long-context comprehension, reduce hallucination, and enhance in-context learning capabilities. We rigorously tested multiple novel architectural components through extensive experimentation to determine the optimal architecture for Motif-2.6B. Comprehensive evaluations demonstrate that Motif-2.6B consistently meets or exceeds the performance of similarly sized state-of-the-art models across diverse benchmarks, showcasing its effectiveness, scalability, and real-world applicability. Through detailed experiments and tailored techniques, Motif-2.6B significantly advances the landscape of efficient, scalable, and powerful foundational LLMs, offering valuable insights and a robust foundation for future research and deployment. 

---
# Agoran: An Agentic Open Marketplace for 6G RAN Automation 

**Authors**: Ilias Chatzistefanidis, Navid Nikaein, Andrea Leone, Ali Maatouk, Leandros Tassioulas, Roberto Morabito, Ioannis Pitsiorlas, Marios Kountouris  

**Link**: [PDF](https://arxiv.org/pdf/2508.09159)  

**Abstract**: Next-generation mobile networks must reconcile the often-conflicting goals of multiple service owners. However, today's network slice controllers remain rigid, policy-bound, and unaware of the business context. We introduce Agoran Service and Resource Broker (SRB), an agentic marketplace that brings stakeholders directly into the operational loop. Inspired by the ancient Greek agora, Agoran distributes authority across three autonomous AI branches: a Legislative branch that answers compliance queries using retrieval-augmented Large Language Models (LLMs); an Executive branch that maintains real-time situational awareness through a watcher-updated vector database; and a Judicial branch that evaluates each agent message with a rule-based Trust Score, while arbitrating LLMs detect malicious behavior and apply real-time incentives to restore trust. Stakeholder-side Negotiation Agents and the SRB-side Mediator Agent negotiate feasible, Pareto-optimal offers produced by a multi-objective optimizer, reaching a consensus intent in a single round, which is then deployed to Open and AI RAN controllers. Deployed on a private 5G testbed and evaluated with realistic traces of vehicle mobility, Agoran achieved significant gains: (i) a 37% increase in throughput of eMBB slices, (ii) a 73% reduction in latency of URLLC slices, and concurrently (iii) an end-to-end 8.3% saving in PRB usage compared to a static baseline. An 1B-parameter Llama model, fine-tuned for five minutes on 100 GPT-4 dialogues, recovers approximately 80% of GPT-4.1's decision quality, while operating within 6 GiB of memory and converging in only 1.3 seconds. These results establish Agoran as a concrete, standards-aligned path toward ultra-flexible, stakeholder-centric 6G networks. A live demo is presented this https URL\&ab_channel=BubbleRAN. 

---
# TEN: Table Explicitization, Neurosymbolically 

**Authors**: Nikita Mehrotra, Aayush Kumar, Sumit Gulwani, Arjun Radhakrishna, Ashish Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2508.09324)  

**Abstract**: We present a neurosymbolic approach, TEN, for extracting tabular data from semistructured input text. This task is particularly challenging for text input that does not use special delimiters consistently to separate columns and rows. Purely neural approaches perform poorly due to hallucinations and their inability to enforce hard constraints. TEN uses Structural Decomposition prompting - a specialized chain-of-thought prompting approach - on a large language model (LLM) to generate an initial table, and thereafter uses a symbolic checker to evaluate not only the well-formedness of that table, but also detect cases of hallucinations or forgetting. The output of the symbolic checker is processed by a critique-LLM to generate guidance for fixing the table, which is presented to the original LLM in a self-debug loop. Our extensive experiments demonstrate that TEN significantly outperforms purely neural baselines across multiple datasets and metrics, achieving significantly higher exact match accuracy and substantially reduced hallucination rates. A 21-participant user study further confirms that TEN's tables are rated significantly more accurate (mean score: 5.0 vs 4.3; p = 0.021), and are consistently preferred for ease of verification and correction, with participants favoring our method in over 60% of the cases. 

---
# Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing 

**Authors**: Xu Wang, Chenkai Xu, Yijie Jin, Jiachun Jin, Hao Zhang, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2508.09192)  

**Abstract**: Diffusion Large Language Models (dLLMs) have emerged as a promising alternative to autoregressive (AR) LLMs for text generation, with the potential to decode multiple tokens in a single iteration. However, none of the existing open-source dLLMs have achieved superior inference speed over AR LLMs of similar size. This paper breaks this barrier based on a simple and effective strategy named discrete diffusion forcing (D2F). D2F equips dLLMs with two key capabilities: (1) block-wise autoregressive generation to enable KV cache utilization; (2) prediction of following tokens without requiring completion of prior blocks for inter-block parallel decoding. In this way, the vanilla dLLMs are refurbished into an AR-diffusion hybrid paradigm for efficient inference. D2F can be implemented with an asymmetric distillation process based on pre-trained dLLMs. We further propose a pipelined parallel decoding algorithm, which enables a trade-off between efficiency and efficacy. Empirically, D2F dLLMs achieve more than $\mathbf{2.5\times}$ inference speed than LLaMA3 and Qwen2.5 on GSM8K. Compared to vanilla dLLMs like LLaDA and Dream, the acceleration can be more than $\mathbf{50\times}$ while maintaining comparable output quality. The code is available at this https URL. 

---
# 5G Core Fault Detection and Root Cause Analysis using Machine Learning and Generative AI 

**Authors**: Joseph H. R. Isaac, Harish Saradagam, Nallamothu Pardhasaradhi  

**Link**: [PDF](https://arxiv.org/pdf/2508.09152)  

**Abstract**: With the advent of 5G networks and technologies, ensuring the integrity and performance of packet core traffic is paramount. During network analysis, test files such as Packet Capture (PCAP) files and log files will contain errors if present in the system that must be resolved for better overall network performance, such as connectivity strength and handover quality. Current methods require numerous person-hours to sort out testing results and find the faults. This paper presents a novel AI/ML-driven Fault Analysis (FA) Engine designed to classify successful and faulty frames in PCAP files, specifically within the 5G packet core. The FA engine analyses network traffic using natural language processing techniques to identify anomalies and inefficiencies, significantly reducing the effort time required and increasing efficiency. The FA Engine also suggests steps to fix the issue using Generative AI via a Large Language Model (LLM) trained on several 5G packet core documents. The engine explains the details of the error from the domain perspective using documents such as the 3GPP standards and user documents regarding the internal conditions of the tests. Test results on the ML models show high classification accuracy on the test dataset when trained with 80-20 splits for the successful and failed PCAP files. Future scopes include extending the AI engine to incorporate 4G network traffic and other forms of network data, such as log text files and multimodal systems. 

---
# Describe What You See with Multimodal Large Language Models to Enhance Video Recommendations 

**Authors**: Marco De Nadai, Andreas Damianou, Mounia Lalmas  

**Link**: [PDF](https://arxiv.org/pdf/2508.09789)  

**Abstract**: Existing video recommender systems rely primarily on user-defined metadata or on low-level visual and acoustic signals extracted by specialised encoders. These low-level features describe what appears on the screen but miss deeper semantics such as intent, humour, and world knowledge that make clips resonate with viewers. For example, is a 30-second clip simply a singer on a rooftop, or an ironic parody filmed amid the fairy chimneys of Cappadocia, Turkey? Such distinctions are critical to personalised recommendations yet remain invisible to traditional encoding pipelines. In this paper, we introduce a simple, recommendation system-agnostic zero-finetuning framework that injects high-level semantics into the recommendation pipeline by prompting an off-the-shelf Multimodal Large Language Model (MLLM) to summarise each clip into a rich natural-language description (e.g. "a superhero parody with slapstick fights and orchestral stabs"), bridging the gap between raw content and user intent. We use MLLM output with a state-of-the-art text encoder and feed it into standard collaborative, content-based, and generative recommenders. On the MicroLens-100K dataset, which emulates user interactions with TikTok-style videos, our framework consistently surpasses conventional video, audio, and metadata features in five representative models. Our findings highlight the promise of leveraging MLLMs as on-the-fly knowledge extractors to build more intent-aware video recommenders. 

---
# TFRank: Think-Free Reasoning Enables Practical Pointwise LLM Ranking 

**Authors**: Yongqi Fan, Xiaoyang Chen, Dezhi Ye, Jie Liu, Haijin Liang, Jin Ma, Ben He, Yingfei Sun, Tong Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2508.09539)  

**Abstract**: Reasoning-intensive ranking models built on Large Language Models (LLMs) have made notable progress, but existing approaches often rely on large-scale LLMs and explicit Chain-of-Thought (CoT) reasoning, resulting in high computational cost and latency that limit real-world use. To address this, we propose \textbf{TFRank}, an efficient pointwise reasoning ranker based on small-scale LLMs. To improve ranking performance, TFRank effectively integrates CoT data, fine-grained score supervision, and multi-task training. Furthermore, it achieves an efficient ``\textbf{T}hink-\textbf{F}ree" reasoning capability by employing a ``think-mode switch'' and pointwise format constraints. Specifically, this allows the model to leverage explicit reasoning during training while delivering precise relevance scores for complex queries at inference without generating any reasoning chains. Experiments show that TFRank (e.g., 1.7B) achieves performance comparable to models with four times more parameters on the BRIGHT benchmark, and demonstrates strong competitiveness on the BEIR benchmark. Further analysis shows that TFRank achieves an effective balance between performance and efficiency, providing a practical solution for integrating advanced reasoning into real-world systems. Our code and data are released in the repository: this https URL. 

---
# Towards Self-cognitive Exploration: Metacognitive Knowledge Graph Retrieval Augmented Generation 

**Authors**: Xujie Yuan, Shimin Di, Jielong Tang, Libin Zheng, Jian Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.09460)  

**Abstract**: Knowledge Graph-based Retrieval-Augmented Generation (KG-RAG) significantly enhances the reasoning capabilities of LargeLanguage Models by leveraging structured knowledge. However, existing KG-RAG frameworks typically operate as open-loop systems, suffering from cognitive blindness, an inability to recognize their exploration deficiencies. This leads to relevance drift and incomplete evidence, which existing self-refinement methods, designed for unstructured text-based RAG, cannot effectively resolve due to the path-dependent nature of graph exploration. To address this challenge, we propose Metacognitive Knowledge Graph Retrieval Augmented Generation (MetaKGRAG), a novel framework inspired by the human metacognition process, which introduces a Perceive-Evaluate-Adjust cycle to enable path-aware, closed-loop refinement. This cycle empowers the system to self-assess exploration quality, identify deficiencies in coverage or relevance, and perform trajectory-connected corrections from precise pivot points. Extensive experiments across five datasets in the medical, legal, and commonsense reasoning domains demonstrate that MetaKGRAG consistently outperforms strong KG-RAG and self-refinement baselines. Our results validate the superiority of our approach and highlight the critical need for path-aware refinement in structured knowledge retrieval. 

---
# Performance of GPT-5 Frontier Models in Ophthalmology Question Answering 

**Authors**: Fares Antaki, David Mikhail, Daniel Milad, Danny A Mammo, Sumit Sharma, Sunil K Srivastava, Bing Yu Chen, Samir Touma, Mertcan Sevgi, Jonathan El-Khoury, Pearse A Keane, Qingyu Chen, Yih Chung Tham, Renaud Duval  

**Link**: [PDF](https://arxiv.org/pdf/2508.09956)  

**Abstract**: Large language models (LLMs) such as GPT-5 integrate advanced reasoning capabilities that may improve performance on complex medical question-answering tasks. For this latest generation of reasoning models, the configurations that maximize both accuracy and cost-efficiency have yet to be established. We evaluated 12 configurations of OpenAI's GPT-5 series (three model tiers across four reasoning effort settings) alongside o1-high, o3-high, and GPT-4o, using 260 closed-access multiple-choice questions from the American Academy of Ophthalmology Basic Clinical Science Course (BCSC) dataset. The primary outcome was multiple-choice accuracy; secondary outcomes included head-to-head ranking via a Bradley-Terry model, rationale quality assessment using a reference-anchored, pairwise LLM-as-a-judge framework, and analysis of accuracy-cost trade-offs using token-based cost estimates. GPT-5-high achieved the highest accuracy (0.965; 95% CI, 0.942-0.985), outperforming all GPT-5-nano variants (P < .001), o1-high (P = .04), and GPT-4o (P < .001), but not o3-high (0.958; 95% CI, 0.931-0.981). GPT-5-high ranked first in both accuracy (1.66x stronger than o3-high) and rationale quality (1.11x stronger than o3-high). Cost-accuracy analysis identified several GPT-5 configurations on the Pareto frontier, with GPT-5-mini-low offering the most favorable low-cost, high-performance balance. These results benchmark GPT-5 on a high-quality ophthalmology dataset, demonstrate the influence of reasoning effort on accuracy, and introduce an autograder framework for scalable evaluation of LLM-generated answers against reference standards in ophthalmology. 

---
# Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks 

**Authors**: Baran Atalar, Eddie Zhang, Carlee Joe-Wong  

**Link**: [PDF](https://arxiv.org/pdf/2508.09958)  

**Abstract**: With the increasing popularity of large language models (LLMs) for a variety of tasks, there has been a growing interest in strategies that can predict which out of a set of LLMs will yield a successful answer at low cost. This problem promises to become more and more relevant as providers like Microsoft allow users to easily create custom LLM "assistants" specialized to particular types of queries. However, some tasks (i.e., queries) may be too specialized and difficult for a single LLM to handle alone. These applications often benefit from breaking down the task into smaller subtasks, each of which can then be executed by a LLM expected to perform well on that specific subtask. For example, in extracting a diagnosis from medical records, one can first select an LLM to summarize the record, select another to validate the summary, and then select another, possibly different, LLM to extract the diagnosis from the summarized record. Unlike existing LLM selection or routing algorithms, this setting requires that we select a sequence of LLMs, with the output of each LLM feeding into the next and potentially influencing its success. Thus, unlike single LLM selection, the quality of each subtask's output directly affects the inputs, and hence the cost and success rate, of downstream LLMs, creating complex performance dependencies that must be learned and accounted for during selection. We propose a neural contextual bandit-based algorithm that trains neural networks that model LLM success on each subtask in an online manner, thus learning to guide the LLM selections for the different subtasks, even in the absence of historical LLM performance data. Experiments on telecommunications question answering and medical diagnosis prediction datasets illustrate the effectiveness of our proposed approach compared to other LLM selection algorithms. 

---
# Transforming Questions and Documents for Semantically Aligned Retrieval-Augmented Generation 

**Authors**: Seokgi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.09755)  

**Abstract**: We introduce a novel retrieval-augmented generation (RAG) framework tailored for multihop question answering. First, our system uses large language model (LLM) to decompose complex multihop questions into a sequence of single-hop subquestions that guide document retrieval. This decomposition mitigates the ambiguity inherent in multi-hop queries by clearly targeting distinct knowledge facets. Second, instead of embedding raw or chunked documents directly, we generate answerable questions from each document chunk using Qwen3-8B, embed these generated questions, and retrieve relevant chunks via question-question embedding similarity. During inference, the retrieved chunks are then fed along with the original question into the RAG pipeline. We evaluate on three multihop question datasets (MuSiQue, 2WikiMultiHopQa, HotpotQA) from LongBench. Our method improves RAG performacne compared to baseline systems. Our contributions highlight the benefits of using answerable-question embeddings for RAG, and the effectiveness of LLM-based query decomposition for multihop scenarios. 

---
# Slow Tuning and Low-Entropy Masking for Safe Chain-of-Thought Distillation 

**Authors**: Ziyang Ma, Qingyue Yuan, Linhai Zhang, Deyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.09666)  

**Abstract**: Previous chain-of-thought (CoT) distillation methods primarily focused on enhancing the reasoning capabilities of Small Language Models (SLMs) by utilizing high-quality rationales generated by powerful Large Language Models (LLMs, e.g., GPT-4). However, few works have noted the negative effects on SLM safety brought by the training, which are revealed in this study. Although there are works on safety alignment that fine-tune language models or manipulate model weights to defend against harmful inputs, they require extra computation or annotated data, and probably impact the reasoning ability of SLMs. In this paper, we investigate how to maintain the safety of SLMs during the CoT distillation process. Specifically, we propose a safe distillation method, Slow Tuning and Low-Entropy Masking Distillation (SLowED), containing two modules: Slow Tuning and Low-Entropy Masking. Slow Tuning scales down the magnitude of model weight changes to optimize the model weights in the neighboring space near the initial weight distribution. Low-Entropy Masking masks low-entropy tokens, which are regarded as unnecessary learning targets, to exclude them from fine-tuning. Experiments on three SLMs (Qwen2.5-1.5B, Llama-3.2-1B, BLOOM-1.1B) across reasoning benchmarks (BBH, BB-Sub, ARC, AGIEval) and safety evaluation (AdvBench) show that SLowED retains the safety of SLMs and comparably improves their reasoning capability compared to existing distillation methods. Furthermore, our ablation study presents the effectiveness of Slow Tuning and Low-Entropy Masking, with the former maintaining the model's safety in the early stage and the latter prolonging the safe training epochs. 

---
# AINL-Eval 2025 Shared Task: Detection of AI-Generated Scientific Abstracts in Russian 

**Authors**: Tatiana Batura, Elena Bruches, Milana Shvenk, Valentin Malykh  

**Link**: [PDF](https://arxiv.org/pdf/2508.09622)  

**Abstract**: The rapid advancement of large language models (LLMs) has revolutionized text generation, making it increasingly difficult to distinguish between human- and AI-generated content. This poses a significant challenge to academic integrity, particularly in scientific publishing and multilingual contexts where detection resources are often limited. To address this critical gap, we introduce the AINL-Eval 2025 Shared Task, specifically focused on the detection of AI-generated scientific abstracts in Russian. We present a novel, large-scale dataset comprising 52,305 samples, including human-written abstracts across 12 diverse scientific domains and AI-generated counterparts from five state-of-the-art LLMs (GPT-4-Turbo, Gemma2-27B, Llama3.3-70B, Deepseek-V3, and GigaChat-Lite). A core objective of the task is to challenge participants to develop robust solutions capable of generalizing to both (i) previously unseen scientific domains and (ii) models not included in the training data. The task was organized in two phases, attracting 10 teams and 159 submissions, with top systems demonstrating strong performance in identifying AI-generated content. We also establish a continuous shared task platform to foster ongoing research and long-term progress in this important area. The dataset and platform are publicly available at this https URL. 

---
# EffiEval: Efficient and Generalizable Model Evaluation via Capability Coverage Maximization 

**Authors**: Yaoning Wang, Jiahao Ying, Yixin Cao, Yubo Ma, Yugang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09662)  

**Abstract**: The rapid advancement of large language models (LLMs) and the development of increasingly large and diverse evaluation benchmarks have introduced substantial computational challenges for model assessment. In this paper, we present EffiEval, a training-free approach for efficient benchmarking that effectively addresses data redundancy while maintaining high evaluation reliability. Our method is specifically designed to meet three key criteria for high-quality evaluation: representativeness, by ensuring comprehensive coverage of model capabilities; fairness, by remaining independent of model performance during sample selection to avoid bias; and generalizability, by enabling flexible transfer across datasets and model families without reliance on large-scale evaluation data. Unlike traditional methods that rely on absolute performance or require extensive evaluation data, our approach adaptively selects high-quality representative subsets based on the Model Utility Index (MUI). Extensive experiments on multiple public benchmarks and diverse LLMs demonstrate that EffiEval achieves strong ranking consistency with full-dataset evaluation using only a small fraction of the original data. Furthermore, our method is flexible and scalable in size, allowing users to balance evaluation efficiency and representativeness according to specific needs. Overall, EffiEval provides a practical and generalizable solution for reliable, fair, and efficient evaluation in the era of LLMs. 

---
# LACA: Improving Cross-lingual Aspect-Based Sentiment Analysis with LLM Data Augmentation 

**Authors**: Jakub Šmíd, Pavel Přibáň, Pavel Král  

**Link**: [PDF](https://arxiv.org/pdf/2508.09515)  

**Abstract**: Cross-lingual aspect-based sentiment analysis (ABSA) involves detailed sentiment analysis in a target language by transferring knowledge from a source language with available annotated data. Most existing methods depend heavily on often unreliable translation tools to bridge the language gap. In this paper, we propose a new approach that leverages a large language model (LLM) to generate high-quality pseudo-labelled data in the target language without the need for translation tools. First, the framework trains an ABSA model to obtain predictions for unlabelled target language data. Next, LLM is prompted to generate natural sentences that better represent these noisy predictions than the original text. The ABSA model is then further fine-tuned on the resulting pseudo-labelled dataset. We demonstrate the effectiveness of this method across six languages and five backbone models, surpassing previous state-of-the-art translation-based approaches. The proposed framework also supports generative models, and we show that fine-tuned LLMs outperform smaller multilingual models. 

---
# Columbo: Expanding Abbreviated Column Names for Tabular Data Using Large Language Models 

**Authors**: Ting Cai, Stephen Sheen, AnHai Doan  

**Link**: [PDF](https://arxiv.org/pdf/2508.09403)  

**Abstract**: Expanding the abbreviated column names of tables, such as ``esal'' to ``employee salary'', is critical for numerous downstream data tasks. This problem arises in enterprises, domain sciences, government agencies, and more. In this paper we make three contributions that significantly advances the state of the art. First, we show that synthetic public data used by prior work has major limitations, and we introduce 4 new datasets in enterprise/science domains, with real-world abbreviations. Second, we show that accuracy measures used by prior work seriously undercount correct expansions, and we propose new synonym-aware measures that capture accuracy much more accurately. Finally, we develop Columbo, a powerful LLM-based solution that exploits context, rules, chain-of-thought reasoning, and token-level analysis. Extensive experiments show that Columbo significantly outperforms NameGuess, the current most advanced solution, by 4-29\%, over 5 datasets. Columbo has been used in production on EDI, a major data portal for environmental sciences. 

---
# Speed Always Wins: A Survey on Efficient Architectures for Large Language Models 

**Authors**: Weigao Sun, Jiaxi Hu, Yucheng Zhou, Jusen Du, Disen Lan, Kexin Wang, Tong Zhu, Xiaoye Qu, Yu Zhang, Xiaoyu Mo, Daizong Liu, Yuxuan Liang, Wenliang Chen, Guoqi Li, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.09834)  

**Abstract**: Large Language Models (LLMs) have delivered impressive results in language understanding, generation, reasoning, and pushes the ability boundary of multimodal models. Transformer models, as the foundation of modern LLMs, offer a strong baseline with excellent scaling properties. However, the traditional transformer architecture requires substantial computations and poses significant obstacles for large-scale training and practical deployment. In this survey, we offer a systematic examination of innovative LLM architectures that address the inherent limitations of transformers and boost the efficiency. Starting from language modeling, this survey covers the background and technical details of linear and sparse sequence modeling methods, efficient full attention variants, sparse mixture-of-experts, hybrid model architectures incorporating the above techniques, and emerging diffusion LLMs. Additionally, we discuss applications of these techniques to other modalities and consider their wider implications for developing scalable, resource-aware foundation models. By grouping recent studies into the above category, this survey presents a blueprint of modern efficient LLM architectures, and we hope this could help motivate future research toward more efficient, versatile AI systems. 

---
# Decoding Neural Emotion Patterns through Natural Language Processing Embeddings 

**Authors**: Gideon Vos, Maryam Ebrahimpour, Liza van Eijk, Zoltan Sarnyai, Mostafa Rahimi Azghadi  

**Link**: [PDF](https://arxiv.org/pdf/2508.09337)  

**Abstract**: Understanding how emotional expression in language relates to brain function is a challenge in computational neuroscience and affective computing. Traditional neuroimaging is costly and lab-bound, but abundant digital text offers new avenues for emotion-brain mapping. Prior work has largely examined neuroimaging-based emotion localization or computational text analysis separately, with little integration. We propose a computational framework that maps textual emotional content to anatomically defined brain regions without requiring neuroimaging. Using OpenAI's text-embedding-ada-002, we generate high-dimensional semantic representations, apply dimensionality reduction and clustering to identify emotional groups, and map them to 18 brain regions linked to emotional processing. Three experiments were conducted: i) analyzing conversational data from healthy vs. depressed subjects (DIAC-WOZ dataset) to compare mapping patterns, ii) applying the method to the GoEmotions dataset and iii) comparing human-written text with large language model (LLM) responses to assess differences in inferred brain activation. Emotional intensity was scored via lexical analysis. Results showed neuroanatomically plausible mappings with high spatial specificity. Depressed subjects exhibited greater limbic engagement tied to negative affect. Discrete emotions were successfully differentiated. LLM-generated text matched humans in basic emotion distribution but lacked nuanced activation in empathy and self-referential regions (medial prefrontal and posterior cingulate cortex). This cost-effective, scalable approach enables large-scale analysis of naturalistic language, distinguishes between clinical populations, and offers a brain-based benchmark for evaluating AI emotional expression. 

---
