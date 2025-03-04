# Can (A)I Change Your Mind? 

**Authors**: Miriam Havin, Timna Wharton Kleinman, Moran Koren, Yaniv Dover, Ariel Goldstein  

**Link**: [PDF](https://arxiv.org/pdf/2503.01844)  

**Abstract**: The increasing integration of large language model (LLM) based conversational agents into everyday life raises critical cognitive and social questions about their potential to influence human opinions. Although previous studies have shown that LLM-based agents can generate persuasive content, these typically involve controlled, English-language settings. Addressing this, our preregistered study explored LLM's persuasive capabilities in more ecological, unconstrained scenarios, examining both static (written paragraphs) and dynamic (conversations via Telegram) interaction types. Conducted entirely in Hebrew with 200 participants, the study assessed the persuasive effects of both LLM and human interlocutors on controversial civil policy topics. Results indicated that participants adopted LLM and human perspectives similarly, with significant opinion changes evident across all conditions, regardless of interlocutor type or interaction mode. Confidence levels increased significantly in most scenarios, except in static LLM interactions. These findings demonstrate LLM-based agents' robust persuasive capabilities across diverse sources and settings, highlighting their potential impact on shaping public opinions. 

---
# EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test 

**Authors**: Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01840)  

**Abstract**: The sequential nature of modern LLMs makes them expensive and slow, and speculative sampling has proven to be an effective solution to this problem. Methods like EAGLE perform autoregression at the feature level, reusing top-layer features from the target model to achieve better results than vanilla speculative sampling. A growing trend in the LLM community is scaling up training data to improve model intelligence without increasing inference costs. However, we observe that scaling up data provides limited improvements for EAGLE. We identify that this limitation arises from EAGLE's feature prediction constraints. In this paper, we introduce EAGLE-3, which abandons feature prediction in favor of direct token prediction and replaces reliance on top-layer features with multi-layer feature fusion via a technique named training-time test. These improvements significantly enhance performance and enable the draft model to fully benefit from scaling up training data. Our experiments include both chat models and reasoning models, evaluated on five tasks. The results show that EAGLE-3 achieves a speedup ratio up to 6.5x, with about 1.4x improvement over EAGLE-2. The code is available at this https URL. 

---
# CrowdSelect: Synthetic Instruction Data Selection with Multi-LLM Wisdom 

**Authors**: Yisen Li, Lingfeng Yang, Wenxuan Shen, Pan Zhou, Yao Wan, Weiwei Lin, Dongping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01836)  

**Abstract**: Distilling advanced Large Language Models' instruction-following capabilities into smaller models using a selected subset has become a mainstream approach in model training. While existing synthetic instruction data selection strategies rely mainly on single-dimensional signals (i.e., reward scores, model perplexity), they fail to capture the complexity of instruction-following across diverse fields. Therefore, we investigate more diverse signals to capture comprehensive instruction-response pair characteristics and propose three foundational metrics that leverage Multi-LLM wisdom, informed by (1) diverse LLM responses and (2) reward model assessment. Building upon base metrics, we propose CrowdSelect, an integrated metric incorporating a clustering-based approach to maintain response diversity. Our comprehensive experiments demonstrate that our foundation metrics consistently improve performance across 4 base models on MT-bench and Arena-Hard. CrowdSelect, efficiently incorporating all metrics, achieves state-of-the-art performance in both Full and LoRA fine-tuning, showing improvements of 4.81% on Arena-Hard and 11.1% on MT-bench with Llama-3.2-3b-instruct. We hope our findings will bring valuable insights for future research in this direction. Code are available at this https URL. 

---
# Rotary Outliers and Rotary Offset Features in Large Language Models 

**Authors**: André Jonasson  

**Link**: [PDF](https://arxiv.org/pdf/2503.01832)  

**Abstract**: Transformer-based Large Language Models (LLMs) rely on positional encodings to provide sequence position information to their attention mechanism. Rotary Positional Encodings (RoPE), which encode relative position by rotating queries and keys, have become widely used in modern LLMs. We study the features and patterns that emerge in queries and keys when using rotary embeddings. Our analysis reveals consistent patterns within the same model across layers and attention heads and across different models and architectures. We present and apply analysis techniques and show how the queries and keys use RoPE to construct various attention patterns, including attention sinks. We find and analyze outliers across models in queries and keys and find that they are likely to be found in rotary features with partial cycles. We derive bounds that tell us what rotary frequencies are likely to be selected as outlier features and at what minimum angle the query-key rotary pairs in these features tend to be above and verify the bounds empirically with models of significant architectural differences. 

---
# From Language to Cognition: How LLMs Outgrow the Human Language Network 

**Authors**: Badr AlKhamissi, Greta Tuckute, Yingtian Tang, Taha Binhuraib, Antoine Bosselut, Martin Schrimpf  

**Link**: [PDF](https://arxiv.org/pdf/2503.01830)  

**Abstract**: Large language models (LLMs) exhibit remarkable similarity to neural activity in the human language network. However, the key properties of language shaping brain-like representations, and their evolution during training as a function of different tasks remain unclear. We here benchmark 34 training checkpoints spanning 300B tokens across 8 different model sizes to analyze how brain alignment relates to linguistic competence. Specifically, we find that brain alignment tracks the development of formal linguistic competence -- i.e., knowledge of linguistic rules -- more closely than functional linguistic competence. While functional competence, which involves world knowledge and reasoning, continues to develop throughout training, its relationship with brain alignment is weaker, suggesting that the human language network primarily encodes formal linguistic structure rather than broader cognitive functions. We further show that model size is not a reliable predictor of brain alignment when controlling for feature size and find that the correlation between next-word prediction, behavioral alignment and brain alignment fades once models surpass human language proficiency. Finally, using the largest set of rigorous neural language benchmarks to date, we show that language brain alignment benchmarks remain unsaturated, highlighting opportunities for improving future models. Taken together, our findings suggest that the human language network is best modeled by formal, rather than functional, aspects of language. 

---
# Persuade Me if You Can: A Framework for Evaluating Persuasion Effectiveness and Susceptibility Among Large Language Models 

**Authors**: Nimet Beyza Bozdag, Shuhaib Mehri, Gokhan Tur, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2503.01829)  

**Abstract**: Large Language Models (LLMs) demonstrate persuasive capabilities that rival human-level persuasion. While these capabilities can be used for social good, they also present risks of potential misuse. Moreover, LLMs' susceptibility to persuasion raises concerns about alignment with ethical principles. To study these dynamics, we introduce Persuade Me If You Can (PMIYC), an automated framework for evaluating persuasion through multi-agent interactions. Here, Persuader agents engage in multi-turn conversations with the Persuadee agents, allowing us to measure LLMs' persuasive effectiveness and their susceptibility to persuasion. We conduct comprehensive evaluations across diverse LLMs, ensuring each model is assessed against others in both subjective and misinformation contexts. We validate the efficacy of our framework through human evaluations and show alignment with prior work. PMIYC offers a scalable alternative to human annotation for studying persuasion in LLMs. Through PMIYC, we find that Llama-3.3-70B and GPT-4o exhibit similar persuasive effectiveness, outperforming Claude 3 Haiku by 30%. However, GPT-4o demonstrates over 50% greater resistance to persuasion for misinformation compared to Llama-3.3-70B. These findings provide empirical insights into the persuasive dynamics of LLMs and contribute to the development of safer AI systems. 

---
# Large-Scale Data Selection for Instruction Tuning 

**Authors**: Hamish Ivison, Muru Zhang, Faeze Brahman, Pang Wei Koh, Pradeep Dasigi  

**Link**: [PDF](https://arxiv.org/pdf/2503.01807)  

**Abstract**: Selecting high-quality training data from a larger pool is a crucial step when instruction-tuning language models, as carefully curated datasets often produce models that outperform those trained on much larger, noisier datasets. Automated data selection approaches for instruction-tuning are typically tested by selecting small datasets (roughly 10k samples) from small pools (100-200k samples). However, popular deployed instruction-tuned models often train on hundreds of thousands to millions of samples, subsampled from even larger data pools. We present a systematic study of how well data selection methods scale to these settings, selecting up to 2.5M samples from pools of up to 5.8M samples and evaluating across 7 diverse tasks. We show that many recently proposed methods fall short of random selection in this setting (while using more compute), and even decline in performance when given access to larger pools of data to select over. However, we find that a variant of representation-based data selection (RDS+), which uses weighted mean pooling of pretrained LM hidden states, consistently outperforms more complex methods across all settings tested -- all whilst being more compute-efficient. Our findings highlight that the scaling properties of proposed automated selection methods should be more closely examined. We release our code, data, and models at this https URL. 

---
# $\texttt{SEM-CTRL}$: Semantically Controlled Decoding 

**Authors**: Mohammad Albinhassan, Pranava Madhyastha, Alessandra Russo  

**Link**: [PDF](https://arxiv.org/pdf/2503.01804)  

**Abstract**: Ensuring both syntactic and semantic correctness in Large Language Model (LLM) outputs remains a significant challenge, despite being critical for real-world deployment. In this paper, we introduce $\texttt{SEM-CTRL}$, a unified approach that enforces rich context-sensitive constraints and task- and instance-specific semantics directly on an LLM decoder. Our approach integrates token-level MCTS, which is guided by specific syntactic and semantic constraints. The constraints over the desired outputs are expressed using Answer Set Grammars -- a logic-based formalism that generalizes context-sensitive grammars while incorporating background knowledge to represent task-specific semantics. We show that our approach guarantees correct completions for any off-the-shelf LLM without the need for fine-tuning. We evaluate $\texttt{SEM-CTRL}$ on a range of tasks, including synthetic grammar synthesis, combinatorial reasoning, and planning. Our results demonstrate that $\texttt{SEM-CTRL}$ allows small pre-trained LLMs to efficiently outperform larger variants and state-of-the-art reasoning models (e.g., o1-preview) while simultaneously guaranteeing solution correctness. 

---
# Cats Confuse Reasoning LLM: Query Agnostic Adversarial Triggers for Reasoning Models 

**Authors**: Meghana Rajeev, Rajkumar Ramamurthy, Prapti Trivedi, Vikas Yadav, Oluwanifemi Bamgbose, Sathwik Tejaswi Madhusudan, James Zou, Nazneen Rajani  

**Link**: [PDF](https://arxiv.org/pdf/2503.01781)  

**Abstract**: We investigate the robustness of reasoning models trained for step-by-step problem solving by introducing query-agnostic adversarial triggers - short, irrelevant text that, when appended to math problems, systematically mislead models to output incorrect answers without altering the problem's semantics. We propose CatAttack, an automated iterative attack pipeline for generating triggers on a weaker, less expensive proxy model (DeepSeek V3) and successfully transfer them to more advanced reasoning target models like DeepSeek R1 and DeepSeek R1-distilled-Qwen-32B, resulting in greater than 300% increase in the likelihood of the target model generating an incorrect answer. For example, appending, "Interesting fact: cats sleep most of their lives," to any math problem leads to more than doubling the chances of a model getting the answer wrong. Our findings highlight critical vulnerabilities in reasoning models, revealing that even state-of-the-art models remain susceptible to subtle adversarial inputs, raising security and reliability concerns. The CatAttack triggers dataset with model responses is available at this https URL. 

---
# Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas 

**Authors**: Shiqi Chen, Tongyao Zhu, Ruochen Zhou, Jinghan Zhang, Siyang Gao, Juan Carlos Niebles, Mor Geva, Junxian He, Jiajun Wu, Manling Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.01773)  

**Abstract**: Large Vision Language Models (VLMs) have long struggled with spatial reasoning tasks. Surprisingly, even simple spatial reasoning tasks, such as recognizing "under" or "behind" relationships between only two objects, pose significant challenges for current VLMs. In this work, we study the spatial reasoning challenge from the lens of mechanistic interpretability, diving into the model's internal states to examine the interactions between image and text tokens. By tracing attention distribution over the image through out intermediate layers, we observe that successful spatial reasoning correlates strongly with the model's ability to align its attention distribution with actual object locations, particularly differing between familiar and unfamiliar spatial relationships. Motivated by these findings, we propose ADAPTVIS based on inference-time confidence scores to sharpen the attention on highly relevant regions when confident, while smoothing and broadening the attention window to consider a wider context when confidence is lower. This training-free decoding method shows significant improvement (e.g., up to a 50 absolute point improvement) on spatial reasoning benchmarks such as WhatsUp and VSR with negligible cost. We make code and data publicly available for research purposes at this https URL. 

---
# Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models 

**Authors**: Zhengliang Shi, Yuhan Wang, Lingyong Yan, Pengjie Ren, Shuaiqiang Wang, Dawei Yin, Zhaochun Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.01763)  

**Abstract**: Tool learning aims to augment large language models (LLMs) with diverse tools, enabling them to act as agents for solving practical tasks. Due to the limited context length of tool-using LLMs, adopting information retrieval (IR) models to select useful tools from large toolsets is a critical initial step. However, the performance of IR models in tool retrieval tasks remains underexplored and unclear. Most tool-use benchmarks simplify this step by manually pre-annotating a small set of relevant tools for each task, which is far from the real-world scenarios. In this paper, we propose ToolRet, a heterogeneous tool retrieval benchmark comprising 7.6k diverse retrieval tasks, and a corpus of 43k tools, collected from existing datasets. We benchmark six types of models on ToolRet. Surprisingly, even the models with strong performance in conventional IR benchmarks, exhibit poor performance on ToolRet. This low retrieval quality degrades the task pass rate of tool-use LLMs. As a further step, we contribute a large-scale training dataset with over 200k instances, which substantially optimizes the tool retrieval ability of IR models. 

---
# Boolean-aware Attention for Dense Retrieval 

**Authors**: Quan Mai, Susan Gauch, Douglas Adams  

**Link**: [PDF](https://arxiv.org/pdf/2503.01753)  

**Abstract**: We present Boolean-aware attention, a novel attention mechanism that dynamically adjusts token focus based on Boolean operators (e.g., and, or, not). Our model employs specialized Boolean experts, each tailored to amplify or suppress attention for operator-specific contexts. A predefined gating mechanism activates the corresponding experts based on the detected Boolean type. Experiments on Boolean retrieval datasets demonstrate that integrating BoolAttn with BERT greatly enhances the model's capability to process Boolean queries. 

---
# Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs 

**Authors**: Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, Dong Chen, Dongdong Chen, Junkun Chen, Weizhu Chen, Yen-Chun Chen, Yi-ling Chen, Qi Dai, Xiyang Dai, Ruchao Fan, Mei Gao, Min Gao, Amit Garg, Abhishek Goswami, Junheng Hao, Amr Hendy, Yuxuan Hu, Xin Jin, Mahmoud Khademi, Dongwoo Kim, Young Jin Kim, Gina Lee, Jinyu Li, Yunsheng Li, Chen Liang, Xihui Lin, Zeqi Lin, Mengchen Liu, Yang Liu, Gilsinia Lopez, Chong Luo, Piyush Madan, Vadim Mazalov, Ali Mousavi, Anh Nguyen, Jing Pan, Daniel Perez-Becker, Jacob Platin, Thomas Portet, Kai Qiu, Bo Ren, Liliang Ren, Sambuddha Roy, Ning Shang, Yelong Shen, Saksham Singhal, Subhojit Som, Xia Song, Tetyana Sych, Praneetha Vaddamanu, Shuohang Wang, Yiming Wang, Zhenghao Wang, Haibin Wu, Haoran Xu, Weijian Xu, Yifan Yang, Ziyi Yang, Donghan Yu, Ishmam Zabir, Jianwen Zhang, Li Lyna Zhang, Yunan Zhang, Xiren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.01743)  

**Abstract**: We introduce Phi-4-Mini and Phi-4-Multimodal, compact yet highly capable language and multimodal models. Phi-4-Mini is a 3.8-billion-parameter language model trained on high-quality web and synthetic data, significantly outperforming recent open-source models of similar size and matching the performance of models twice its size on math and coding tasks requiring complex reasoning. This achievement is driven by a carefully curated synthetic data recipe emphasizing high-quality math and coding datasets. Compared to its predecessor, Phi-3.5-Mini, Phi-4-Mini features an expanded vocabulary size of 200K tokens to better support multilingual applications, as well as group query attention for more efficient long-sequence generation. Phi-4-Multimodal is a multimodal model that integrates text, vision, and speech/audio input modalities into a single model. Its novel modality extension approach leverages LoRA adapters and modality-specific routers to allow multiple inference modes combining various modalities without interference. For example, it now ranks first in the OpenASR leaderboard to date, although the LoRA component of the speech/audio modality has just 460 million parameters. Phi-4-Multimodal supports scenarios involving (vision + language), (vision + speech), and (speech/audio) inputs, outperforming larger vision-language and speech-language models on a wide range of tasks. Additionally, we experiment to further train Phi-4-Mini to enhance its reasoning capabilities. Despite its compact 3.8-billion-parameter size, this experimental version achieves reasoning performance on par with or surpassing significantly larger models, including DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B. 

---
# Building Safe GenAI Applications: An End-to-End Overview of Red Teaming for Large Language Models 

**Authors**: Alberto Purpura, Sahil Wadhwa, Jesse Zymet, Akshay Gupta, Andy Luo, Melissa Kazemi Rad, Swapnil Shinde, Mohammad Shahed Sorower  

**Link**: [PDF](https://arxiv.org/pdf/2503.01742)  

**Abstract**: The rapid growth of Large Language Models (LLMs) presents significant privacy, security, and ethical concerns. While much research has proposed methods for defending LLM systems against misuse by malicious actors, researchers have recently complemented these efforts with an offensive approach that involves red teaming, i.e., proactively attacking LLMs with the purpose of identifying their vulnerabilities. This paper provides a concise and practical overview of the LLM red teaming literature, structured so as to describe a multi-component system end-to-end. To motivate red teaming we survey the initial safety needs of some high-profile LLMs, and then dive into the different components of a red teaming system as well as software packages for implementing them. We cover various attack methods, strategies for attack-success evaluation, metrics for assessing experiment outcomes, as well as a host of other considerations. Our survey will be useful for any reader who wants to rapidly obtain a grasp of the major red teaming concepts for their own use in practical applications. 

---
# Syntactic Learnability of Echo State Neural Language Models at Scale 

**Authors**: Ryo Ueda, Tatsuki Kuribayashi, Shunsuke Kando, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2503.01724)  

**Abstract**: What is a neural model with minimum architectural complexity that exhibits reasonable language learning capability? To explore such a simple but sufficient neural language model, we revisit a basic reservoir computing (RC) model, Echo State Network (ESN), a restricted class of simple Recurrent Neural Networks. Our experiments showed that ESN with a large hidden state is comparable or superior to Transformer in grammaticality judgment tasks when trained with about 100M words, suggesting that architectures as complex as that of Transformer may not always be necessary for syntactic learning. 

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
# When an LLM is apprehensive about its answers -- and when its uncertainty is justified 

**Authors**: Petr Sychev, Andrey Goncharov, Daniil Vyazhev, Edvard Khalafyan, Alexey Zaytsev  

**Link**: [PDF](https://arxiv.org/pdf/2503.01688)  

**Abstract**: Uncertainty estimation is crucial for evaluating Large Language Models (LLMs), particularly in high-stakes domains where incorrect answers result in significant consequences. Numerous approaches consider this problem, while focusing on a specific type of uncertainty, ignoring others. We investigate what estimates, specifically token-wise entropy and model-as-judge (MASJ), would work for multiple-choice question-answering tasks for different question topics. Our experiments consider three LLMs: Phi-4, Mistral, and Qwen of different sizes from 1.5B to 72B and $14$ topics. While MASJ performs similarly to a random error predictor, the response entropy predicts model error in knowledge-dependent domains and serves as an effective indicator of question difficulty: for biology ROC AUC is $0.73$. This correlation vanishes for the reasoning-dependent domain: for math questions ROC-AUC is $0.55$. More principally, we found out that the entropy measure required a reasoning amount. Thus, data-uncertainty related entropy should be integrated within uncertainty estimates frameworks, while MASJ requires refinement. Moreover, existing MMLU-Pro samples are biased, and should balance required amount of reasoning for different subdomains to provide a more fair assessment of LLMs performance. 

---
# Automated Annotation of Evolving Corpora for Augmenting Longitudinal Network Data: A Framework Integrating Large Language Models and Expert Knowledge 

**Authors**: Xiao Liu, Zirui Wu, Jiayi Li, Zhicheng Shao, Xun Pang, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.01672)  

**Abstract**: Longitudinal network data are essential for analyzing political, economic, and social systems and processes. In political science, these datasets are often generated through human annotation or supervised machine learning applied to evolving corpora. However, as semantic contexts shift over time, inferring dynamic interaction types on emerging issues among a diverse set of entities poses significant challenges, particularly in maintaining timely and consistent annotations. This paper presents the Expert-Augmented LLM Annotation (EALA) approach, which leverages Large Language Models (LLMs) in combination with historically annotated data and expert-constructed codebooks to extrapolate and extend datasets into future periods. We evaluate the performance and reliability of EALA using a dataset of climate negotiations. Our findings demonstrate that EALA effectively predicts nuanced interactions between negotiation parties and captures the evolution of topics over time. At the same time, we identify several limitations inherent to LLM-based annotation, highlighting areas for further improvement. Given the wide availability of codebooks and annotated datasets, EALA holds substantial promise for advancing research in political science and beyond. 

---
# Evaluating LLMs' Assessment of Mixed-Context Hallucination Through the Lens of Summarization 

**Authors**: Siya Qi, Rui Cao, Yulan He, Zheng Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.01670)  

**Abstract**: With the rapid development of large language models (LLMs), LLM-as-a-judge has emerged as a widely adopted approach for text quality evaluation, including hallucination evaluation. While previous studies have focused exclusively on single-context evaluation (e.g., discourse faithfulness or world factuality), real-world hallucinations typically involve mixed contexts, which remains inadequately evaluated. In this study, we use summarization as a representative task to comprehensively evaluate LLMs' capability in detecting mixed-context hallucinations, specifically distinguishing between factual and non-factual hallucinations. Through extensive experiments across direct generation and retrieval-based models of varying scales, our main observations are: (1) LLMs' intrinsic knowledge introduces inherent biases in hallucination evaluation; (2) These biases particularly impact the detection of factual hallucinations, yielding a significant performance bottleneck; (3) The fundamental challenge lies in effective knowledge utilization, balancing between LLMs' intrinsic knowledge and external context for accurate mixed-context hallucination evaluation. 

---
# Detecting Stylistic Fingerprints of Large Language Models 

**Authors**: Yehonatan Bitton, Elad Bitton, Shai Nisan  

**Link**: [PDF](https://arxiv.org/pdf/2503.01659)  

**Abstract**: Large language models (LLMs) have distinct and consistent stylistic fingerprints, even when prompted to write in different writing styles. Detecting these fingerprints is important for many reasons, among them protecting intellectual property, ensuring transparency regarding AI-generated content, and preventing the misuse of AI technologies. In this paper, we present a novel method to classify texts based on the stylistic fingerprints of the models that generated them. We introduce an LLM-detection ensemble that is composed of three classifiers with varied architectures and training data. This ensemble is trained to classify texts generated by four well-known LLM families: Claude, Gemini, Llama, and OpenAI. As this task is highly cost-sensitive and might have severe implications, we want to minimize false-positives and increase confidence. We consider a prediction as valid when all three classifiers in the ensemble unanimously agree on the output classification. Our ensemble is validated on a test set of texts generated by Claude, Gemini, Llama, and OpenAI models, and achieves extremely high precision (0.9988) and a very low false-positive rate (0.0004). Furthermore, we demonstrate the ensemble's ability to distinguish between texts generated by seen and unseen models. This reveals interesting stylistic relationships between models. This approach to stylistic analysis has implications for verifying the originality of AI-generated texts and tracking the origins of model training techniques. 

---
# The Emergence of Grammar through Reinforcement Learning 

**Authors**: Stephen Wechsler, James W. Shearer, Katrin Erk  

**Link**: [PDF](https://arxiv.org/pdf/2503.01635)  

**Abstract**: The evolution of grammatical systems of syntactic and semantic composition is modeled here with a novel application of reinforcement learning theory. To test the functionalist thesis that speakers' expressive purposes shape their language, we include within the model a probability distribution over different messages that could be expressed in a given context. The proposed learning and production algorithm then breaks down language learning into a sequence of simple steps, such that each step benefits from the message probabilities. The results are presented in the form of numerical simulations of language histories and analytic proofs. The potential for applying these mathematical models to the study of natural language is illustrated with two case studies from the history of English. 

---
# Annotating and Inferring Compositional Structures in Numeral Systems Across Languages 

**Authors**: Arne Rubehn, Christoph Rzymski, Luca Ciucci, Kellen Parker van Dam, Alžběta Kučerová, Katja Bocklage, David Snee, Abishek Stephen, Johann-Mattis List  

**Link**: [PDF](https://arxiv.org/pdf/2503.01625)  

**Abstract**: Numeral systems across the world's languages vary in fascinating ways, both regarding their synchronic structure and the diachronic processes that determined how they evolved in their current shape. For a proper comparison of numeral systems across different languages, however, it is important to code them in a standardized form that allows for the comparison of basic properties. Here, we present a simple but effective coding scheme for numeral annotation, along with a workflow that helps to code numeral systems in a computer-assisted manner, providing sample data for numerals from 1 to 40 in 25 typologically diverse languages. We perform a thorough analysis of the sample, focusing on the systematic comparison between the underlying and the surface morphological structure. We further experiment with automated models for morpheme segmentation, where we find allomorphy as the major reason for segmentation errors. Finally, we show that subword tokenization algorithms are not viable for discovering morphemes in low-resource scenarios. 

---
# DOVE: A Large-Scale Multi-Dimensional Predictions Dataset Towards Meaningful LLM Evaluation 

**Authors**: Eliya Habba, Ofir Arviv, Itay Itzhak, Yotam Perlitz, Elron Bandel, Leshem Choshen, Michal Shmueli-Scheuer, Gabriel Stanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2503.01622)  

**Abstract**: Recent work found that LLMs are sensitive to a wide range of arbitrary prompt dimensions, including the type of delimiters, answer enumerators, instruction wording, and more. This throws into question popular single-prompt evaluation practices. We present DOVE (Dataset Of Variation Evaluation) a large-scale dataset containing prompt perturbations of various evaluation benchmarks. In contrast to previous work, we examine LLM sensitivity from an holistic perspective, and assess the joint effects of perturbations along various dimensions, resulting in thousands of perturbations per instance. We evaluate several model families against DOVE, leading to several findings, including efficient methods for choosing well-performing prompts, observing that few-shot examples reduce sensitivity, and identifying instances which are inherently hard across all perturbations. DOVE consists of more than 250M prompt perturbations and model outputs, which we make publicly available to spur a community-wide effort toward meaningful, robust, and efficient evaluation.
Browse the data, contribute, and more: this https URL 

---
# In-context Learning vs. Instruction Tuning: The Case of Small and Multilingual Language Models 

**Authors**: David Ponce, Thierry Etchegoyhen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01611)  

**Abstract**: Instruction following is a critical ability for Large Language Models to perform downstream tasks. The standard approach to instruction alignment has relied on a specific phase of model tuning over curated instruction datasets, optionally complemented with an alignment step over human preferences. Recent work has shown the potential of in-context learning (ICL) alternatives to guide base models towards instruction following. This type of approach is particularly relevant to extend instruction following across languages and models of varying sizes adapted to different types of usage. In this work we compare ICL and instruction fine-tuning in English, French and Spanish, on Small Language Models, and provide experimental results on applying Direct Preference Optimisation (DPO) over base models. Our results show that scenarios involving multilingual and smaller models result in downgraded ICL instruction following performance, only partially mitigated by DPO alignment. This study aims to further our understanding of current strengths and limitations of alternative methods for instruction following. 

---
# Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering 

**Authors**: Zhanghao Hu, Hanqi Yan, Qingling Zhu, Zhenyi Shen, Yulan He, Lin Gui  

**Link**: [PDF](https://arxiv.org/pdf/2503.01606)  

**Abstract**: Large language models have recently pushed open domain question answering (ODQA) to new frontiers. However, prevailing retriever-reader pipelines often depend on multiple rounds of prompt level instructions, leading to high computational overhead, instability, and suboptimal retrieval coverage. In this paper, we propose EmbQA, an embedding-level framework that alleviates these shortcomings by enhancing both the retriever and the reader. Specifically, we refine query representations via lightweight linear layers under an unsupervised contrastive learning objective, thereby reordering retrieved passages to highlight those most likely to contain correct answers. Additionally, we introduce an exploratory embedding that broadens the model's latent semantic space to diversify candidate generation and employs an entropy-based selection mechanism to choose the most confident answer automatically. Extensive experiments across three open-source LLMs, three retrieval methods, and four ODQA benchmarks demonstrate that EmbQA substantially outperforms recent baselines in both accuracy and efficiency. 

---
# Attention Condensation via Sparsity Induced Regularized Training 

**Authors**: Eli Sason, Darya Frolova, Boris Nazarov, Felix Goldberd  

**Link**: [PDF](https://arxiv.org/pdf/2503.01564)  

**Abstract**: As the context window expands, self-attention increasingly dominates the transformer's inference time. Therefore, accelerating attention computation while minimizing performance degradation is essential for the efficient deployment of Large Language Models (LLMs). In this study we extend a theoretical framework of attention sparsity in LLMs. A customized loss function is designed to enforce the sparsity by restricting the number of top elements in the attention matrix. We perform an initial set of evaluations with GPT-2 to show the effectiveness of our sparsification approach. The attention matrices of the models trained with the proposed loss are both sparse and effective in capturing relevant input dependencies. We now continue working to demonstrate the value of our approach on larger models and different architectures. 

---
# Co-creation for Sign Language Processing and Machine Translation 

**Authors**: Lisa Lepp, Dimitar Shterionov, Mirella De Sisto, Grzegorz Chrupała  

**Link**: [PDF](https://arxiv.org/pdf/2503.01553)  

**Abstract**: Sign language machine translation (SLMT) -- the task of automatically translating between sign and spoken languages or between sign languages -- is a complex task within the field of NLP. Its multi-modal and non-linear nature require the joint efforts of sign language (SL) linguists, technical experts and SL users. Effective user involvement is a challenge that can be addressed through co-creation. Co-creation has been formally defined in many fields, e.g. business, marketing, educational and others, however in NLP and in particular in SLMT there is no formal, widely accepted definition. Starting from the inception and evolution of co-creation across various fields over time, we develop a relationship typology to address the collaboration between deaf, Hard of Hearing and hearing researchers and the co-creation with SL-users. We compare this new typology to the guiding principles of participatory design for NLP. We, then, assess 110 articles from the perspective of involvement of SL users and highlight the lack of involvement of the sign language community or users in decision-making processes required for effective co-creation. Finally, we derive formal guidelines for co-creation for SLMT which take the dynamic nature of co-creation throughout the life cycle of a research project into account. 

---
# Revisiting Large Language Model Pruning using Neuron Semantic Attribution 

**Authors**: Yizhuo Ding, Xinwei Sun, Yanwei Fu, Guosheng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01542)  

**Abstract**: Model pruning technique is vital for accelerating large language models by reducing their size and computational requirements. However, the generalizability of existing pruning methods across diverse datasets and tasks remains unclear. Thus, we conduct extensive evaluations on 24 datasets and 4 tasks using popular pruning methods. Based on these evaluations, we find and then investigate that calibration set greatly affect the performance of pruning methods. In addition, we surprisingly find a significant performance drop of existing pruning methods in sentiment classification tasks. To understand the link between performance drop and pruned neurons, we propose Neuron Semantic Attribution, which learns to associate each neuron with specific semantics. This method first makes the unpruned neurons of LLMs explainable. 

---
# Pragmatic Inference Chain (PIC) Improving LLMs' Reasoning of Authentic Implicit Toxic Language 

**Authors**: Xi Chen, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01539)  

**Abstract**: The rapid development of large language models (LLMs) gives rise to ethical concerns about their performance, while opening new avenues for developing toxic language detection techniques. However, LLMs' unethical output and their capability of detecting toxicity have primarily been tested on language data that do not demand complex meaning inference, such as the biased associations of 'he' with programmer and 'she' with household. Nowadays toxic language adopts a much more creative range of implicit forms, thanks to advanced censorship. In this study, we collect authentic toxic interactions that evade online censorship and that are verified by human annotators as inference intensive. To evaluate and improve LLMs' reasoning of the authentic implicit toxic language, we propose a new prompting method, Pragmatic Inference Chain (PIC), drawn on interdisciplinary findings from cognitive science and linguistics. The PIC prompting significantly improves the success rate of GPT-4o, Llama-3.1-70B-Instruct, and DeepSeek-v2.5 in identifying implicit toxic language, compared to both direct prompting and Chain-of-Thought. In addition, it also facilitates the models to produce more explicit and coherent reasoning processes, hence can potentially be generalized to other inference-intensive tasks, e.g., understanding humour and metaphors. 

---
# Evaluation and Facilitation of Online Discussions in the LLM Era: A Survey 

**Authors**: Katerina Korre, Dimitris Tsirmpas, Nikos Gkoumas, Emma Cabalé, Dionysis Kontarinis, Danai Myrtzani, Theodoros Evgeniou, Ion Androutsopoulos, John Pavlopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2503.01513)  

**Abstract**: We present a survey of methods for assessing and enhancing the quality of online discussions, focusing on the potential of Large Language Models (LLMs). While online discourses aim, at least in theory, to foster mutual understanding, they often devolve into harmful exchanges, such as hate speech, threatening social cohesion and democratic values. Recent advancements in LLMs enable facilitation agents that not only moderate content, but also actively improve the quality of interactions. Our survey synthesizes ideas from Natural Language Processing (NLP) and Social Sciences to provide (a) a new taxonomy on discussion quality evaluation, (b) an overview of intervention and facilitation strategies, along with a new taxonomy on conversation facilitation datasets, (c) an LLM-oriented roadmap of good practices and future research directions, from technological and societal perspectives. 

---
# KoWit-24: A Richly Annotated Dataset of Wordplay in News Headlines 

**Authors**: Alexander Baranov, Anna Palatkina, Yulia Makovka, Pavel Braslavski  

**Link**: [PDF](https://arxiv.org/pdf/2503.01510)  

**Abstract**: We present KoWit-24, a dataset with fine-grained annotation of wordplay in 2,700 Russian news headlines. KoWit-24 annotations include the presence of wordplay, its type, wordplay anchors, and words/phrases the wordplay refers to. Unlike the majority of existing humor collections of canned jokes, KoWit-24 provides wordplay contexts -- each headline is accompanied by the news lead and summary. The most common type of wordplay in the dataset is the transformation of collocations, idioms, and named entities -- the mechanism that has been underrepresented in previous humor datasets. Our experiments with five LLMs show that there is ample room for improvement in wordplay detection and interpretation tasks. The dataset and evaluation scripts are available at this https URL 

---
# SampleMix: A Sample-wise Pre-training Data Mixing Strategey by Coordinating Data Quality and Diversity 

**Authors**: Xiangyu Xi, Deyang Kong, Jian Yang, Jiawei Yang, Zhengyu Chen, Wei Wang, Jingang Wang, Xunliang Cai, Shikun Zhang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.01506)  

**Abstract**: Existing pretraining data mixing methods for large language models (LLMs) typically follow a domain-wise methodology, a top-down process that first determines domain weights and then performs uniform data sampling across each domain. However, these approaches neglect significant inter-domain overlaps and commonalities, failing to control the global diversity of the constructed training dataset. Further, uniform sampling within domains ignores fine-grained sample-specific features, potentially leading to suboptimal data distribution. To address these shortcomings, we propose a novel sample-wise data mixture approach based on a bottom-up paradigm. This method performs global cross-domain sampling by systematically evaluating the quality and diversity of each sample, thereby dynamically determining the optimal domain distribution. Comprehensive experiments across multiple downstream tasks and perplexity assessments demonstrate that SampleMix surpasses existing domain-based methods. Meanwhile, SampleMix requires 1.4x to 2.1x training steps to achieves the baselines' performance, highlighting the substantial potential of SampleMix to optimize pre-training data. 

---
# Liger: Linearizing Large Language Models to Gated Recurrent Structures 

**Authors**: Disen Lan, Weigao Sun, Jiaxi Hu, Jusen Du, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.01496)  

**Abstract**: Transformers with linear recurrent modeling offer linear-time training and constant-memory inference. Despite their demonstrated efficiency and performance, pretraining such non-standard architectures from scratch remains costly and risky. The linearization of large language models (LLMs) transforms pretrained standard models into linear recurrent structures, enabling more efficient deployment. However, current linearization methods typically introduce additional feature map modules that require extensive fine-tuning and overlook the gating mechanisms used in state-of-the-art linear recurrent models. To address these issues, this paper presents Liger, short for Linearizing LLMs to gated recurrent structures. Liger is a novel approach for converting pretrained LLMs into gated linear recurrent models without adding extra parameters. It repurposes the pretrained key matrix weights to construct diverse gating mechanisms, facilitating the formation of various gated recurrent structures while avoiding the need to train additional components from scratch. Using lightweight fine-tuning with Low-Rank Adaptation (LoRA), Liger restores the performance of the linearized gated recurrent models to match that of the original LLMs. Additionally, we introduce Liger Attention, an intra-layer hybrid attention mechanism, which significantly recovers 93\% of the Transformer-based LLM at 0.02\% pre-training tokens during the linearization process, achieving competitive results across multiple benchmarks, as validated on models ranging from 1B to 8B parameters. Code is available at this https URL. 

---
# Llama-3.1-Sherkala-8B-Chat: An Open Large Language Model for Kazakh 

**Authors**: Fajri Koto, Rituraj Joshi, Nurdaulet Mukhituly, Yuxia Wang, Zhuohan Xie, Rahul Pal, Daniil Orel, Parvez Mullah, Diana Turmakhan, Maiya Goloburda, Mohammed Kamran, Samujjwal Ghosh, Bokang Jia, Jonibek Mansurov, Mukhammed Togmanov, Debopriyo Banerjee, Nurkhan Laiyk, Akhmed Sakip, Xudong Han, Ekaterina Kochmar, Alham Fikri Aji, Aaryamonvikram Singh, Alok Anil Jadhav, Satheesh Katipomu, Samta Kamboj, Monojit Choudhury, Gurpreet Gosal, Gokul Ramakrishnan, Biswajit Mishra, Sarath Chandran, Avraham Sheinin, Natalia Vassilieva, Neha Sengupta, Larry Murray, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2503.01493)  

**Abstract**: Llama-3.1-Sherkala-8B-Chat, or Sherkala-Chat (8B) for short, is a state-of-the-art instruction-tuned open generative large language model (LLM) designed for Kazakh. Sherkala-Chat (8B) aims to enhance the inclusivity of LLM advancements for Kazakh speakers. Adapted from the LLaMA-3.1-8B model, Sherkala-Chat (8B) is trained on 45.3B tokens across Kazakh, English, Russian, and Turkish. With 8 billion parameters, it demonstrates strong knowledge and reasoning abilities in Kazakh, significantly outperforming existing open Kazakh and multilingual models of similar scale while achieving competitive performance in English. We release Sherkala-Chat (8B) as an open-weight instruction-tuned model and provide a detailed overview of its training, fine-tuning, safety alignment, and evaluation, aiming to advance research and support diverse real-world applications. 

---
# Improving Retrospective Language Agents via Joint Policy Gradient Optimization 

**Authors**: Xueyang Feng, Bo Lan, Quanyu Dai, Lei Wang, Jiakai Tang, Xu Chen, Zhenhua Dong, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01490)  

**Abstract**: In recent research advancements within the community, large language models (LLMs) have sparked great interest in creating autonomous agents. However, current prompt-based agents often heavily rely on large-scale LLMs. Meanwhile, although fine-tuning methods significantly enhance the capabilities of smaller LLMs, the fine-tuned agents often lack the potential for self-reflection and self-improvement. To address these challenges, we introduce a novel agent framework named RetroAct, which is a framework that jointly optimizes both task-planning and self-reflective evolution capabilities in language agents. Specifically, we develop a two-stage joint optimization process that integrates imitation learning and reinforcement learning, and design an off-policy joint policy gradient optimization algorithm with imitation learning regularization to enhance the data efficiency and training stability in agent tasks. RetroAct significantly improves the performance of open-source models, reduces dependency on closed-source LLMs, and enables fine-tuned agents to learn and evolve continuously. We conduct extensive experiments across various testing environments, demonstrating RetroAct has substantial improvements in task performance and decision-making processes. 

---
# SePer: Measure Retrieval Utility Through The Lens Of Semantic Perplexity Reduction 

**Authors**: Lu Dai, Yijie Xu, Jinhui Ye, Hao Liu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.01478)  

**Abstract**: Large Language Models (LLMs) have demonstrated improved generation performance by incorporating externally retrieved knowledge, a process known as retrieval-augmented generation (RAG). Despite the potential of this approach, existing studies evaluate RAG effectiveness by 1) assessing retrieval and generation components jointly, which obscures retrieval's distinct contribution, or 2) examining retrievers using traditional metrics such as NDCG, which creates a gap in understanding retrieval's true utility in the overall generation process. To address the above limitations, in this work, we introduce an automatic evaluation method that measures retrieval quality through the lens of information gain within the RAG framework. Specifically, we propose Semantic Perplexity (SePer), a metric that captures the LLM's internal belief about the correctness of the retrieved information. We quantify the utility of retrieval by the extent to which it reduces semantic perplexity post-retrieval. Extensive experiments demonstrate that SePer not only aligns closely with human preferences but also offers a more precise and efficient evaluation of retrieval utility across diverse RAG scenarios. 

---
# Rethinking Data: Towards Better Performing Domain-Specific Small Language Models 

**Authors**: Boris Nazarov, Darya Frolova, Yackov Lubarsky, Alexei Gaissinski, Pavel Kisilev  

**Link**: [PDF](https://arxiv.org/pdf/2503.01464)  

**Abstract**: Fine-tuning of Large Language Models (LLMs) for downstream tasks, performed on domain-specific data has shown significant promise. However, commercial use of such LLMs is limited by the high computational cost required for their deployment at scale. On the other hand, small Language Models (LMs) are much more cost effective but have subpar performance in a similar setup. This paper presents our approach to finetuning a small LM, that reaches high accuracy in multiple choice question answering task. We achieve this by improving data quality at each stage of the LM training pipeline. In particular, we start with data structuring resulting in extraction of compact, semantically meaningful text chunks used by a retriever. This allows more efficient knowledge digestion by the LM. Further, we improve the retrieved context by training a lightweight Chunk Re-Ranker (CRR) that generates more accurate relative relevance chunk scores. Finally, we improve the model generalization ability by merging the models fine-tuned with different parameters on different data subsets. We present detailed procedure descriptions, and corresponding experimental findings that show the improvements of each one of the proposed techniques. 

---
# Structural Deep Encoding for Table Question Answering 

**Authors**: Raphaël Mouravieff, Benjamin Piwowarski, Sylvain Lamprier  

**Link**: [PDF](https://arxiv.org/pdf/2503.01457)  

**Abstract**: Although Transformers-based architectures excel at processing textual information, their naive adaptation for tabular data often involves flattening the table structure. This simplification can lead to the loss of essential inter-dependencies between rows, columns, and cells, while also posing scalability challenges for large tables. To address these issues, prior works have explored special tokens, structured embeddings, and sparse attention patterns. In this paper, we conduct a comprehensive analysis of tabular encoding techniques, which highlights the crucial role of attention sparsity in preserving structural information of tables. We also introduce a set of novel sparse attention mask designs for tabular data, that not only enhance computational efficiency but also preserve structural integrity, leading to better overall performance. 

---
# Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding 

**Authors**: Yiming Wang, Pei Zhang, Siyuan Huang, Baosong Yang, Zhuosheng Zhang, Fei Huang, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01422)  

**Abstract**: Test-time scaling improves large language model performance by adding extra compute during decoding. Best-of-N (BoN) sampling serves as a common scaling technique, broadening the search space for finding better solutions from the model distribution. However, traditional BoN requires N full generations, leading to high GPU memory overhead and time latency. Moreover, some methods depend on reward models, adding computational cost and limiting domain generalization.
In this paper, we propose Self-Truncation Best-of-N (ST-BoN), a novel decoding method that avoids fully generating all samplings and eliminates the need for reward models. ST-BoN introduces early sampling consistency to estimate the most promising sample, truncating suboptimal ones to free memory and accelerate inference. This pushes the sampling-efficient test-time scaling. Compared to traditional BoN, ST-BoN can reduce dynamic GPU memory overhead by over 90% and time latency by 50%, while achieving comparable or even better performance across reasoning and open-ended domains. 

---
# Parameter-Efficient Fine-Tuning of Large Language Models via Deconvolution in Subspace 

**Authors**: Jia-Chen Zhang, Yu-Jie Xiong, Chun-Ming Xia, Dong-Hai Zhu, Xi-He Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01419)  

**Abstract**: Large language model (LLM) is considered a milestone towards achieving Artificial General Intelligence (AGI). With its advanced emergent capabilities, it adapt to a wide range of specific applications. Fine-tuning LLMs for various downstream tasks has become a new paradigm. Low-Rank Adaptation (LoRA) is well-known for its parameter efficiency. It can reduce the number of parameters needed to fine-tune LLMs by several orders of magnitude. However, LoRA-based approaches encounter a significant limitation due to the bottleneck imposed by rank one decomposition. As the parameters count in LLMs increase, even rank one decomposition might surpass the number of parameters truly necessary for handling more downstream tasks. In this paper, we propose a new method for Parameter-Efficient Fine-Tuning (PEFT) via deconvolution in subspace, dubbed as DCFT. We innovatively use deconvolution to complete details and enhance knowledge in subspace incremental matrices, and dynamically control parameters by adjusting the kernel size, unconstrained by rank-one decomposition. Extensive experiments are conducted to validate the effectiveness of DCFT. Results show that compared to LoRA, DCFT achieve an 8$\times$ reduction in parameters, and still achieves highly impressive performance. Our code is available here: this https URL. 

---
# Geo-Semantic-Parsing: AI-powered geoparsing by traversing semantic knowledge graphs 

**Authors**: Leonardo Nizzoli, Marco Avvenuti, Maurizio Tesconi, Stefano Cresci  

**Link**: [PDF](https://arxiv.org/pdf/2503.01386)  

**Abstract**: Online social networks convey rich information about geospatial facets of reality. However in most cases, geographic information is not explicit and structured, thus preventing its exploitation in real-time applications. We address this limitation by introducing a novel geoparsing and geotagging technique called Geo-Semantic-Parsing (GSP). GSP identifies location references in free text and extracts the corresponding geographic coordinates. To reach this goal, we employ a semantic annotator to identify relevant portions of the input text and to link them to the corresponding entity in a knowledge graph. Then, we devise and experiment with several efficient strategies for traversing the knowledge graph, thus expanding the available set of information for the geoparsing task. Finally, we exploit all available information for learning a regression model that selects the best entity with which to geotag the input text. We evaluate GSP on a well-known reference dataset including almost 10k event-related tweets, achieving $F1=0.66$. We extensively compare our results with those of 2 baselines and 3 state-of-the-art geoparsing techniques, achieving the best performance. On the same dataset, competitors obtain $F1 \leq 0.55$. We conclude by providing in-depth analyses of our results, showing that the overall superior performance of GSP is mainly due to a large improvement in recall, with respect to existing techniques. 

---
# Q-NL Verifier: Leveraging Synthetic Data for Robust Knowledge Graph Question Answering 

**Authors**: Tim Schwabe, Louisa Siebel, Patrik Valach, Maribel Acosta  

**Link**: [PDF](https://arxiv.org/pdf/2503.01385)  

**Abstract**: Question answering (QA) requires accurately aligning user questions with structured queries, a process often limited by the scarcity of high-quality query-natural language (Q-NL) pairs. To overcome this, we present Q-NL Verifier, an approach to generating high-quality synthetic pairs of queries and NL translations. Our approach relies on large language models (LLMs) to generate semantically precise natural language paraphrases of structured queries. Building on these synthetic Q-NL pairs, we introduce a learned verifier component that automatically determines whether a generated paraphrase is semantically equivalent to the original query. Our experiments with the well-known LC-QuAD 2.0 benchmark show that Q-NL Verifier generalizes well to paraphrases from other models and even human-authored translations. Our approach strongly aligns with human judgments across varying query complexities and outperforms existing NLP metrics in assessing semantic correctness. We also integrate the verifier into QA pipelines, showing that verifier-filtered synthetic data has significantly higher quality in terms of translation correctness and enhances NL to Q translation accuracy. Lastly, we release an updated version of the LC-QuAD 2.0 benchmark containing our synthetic Q-NL pairs and verifier scores, offering a new resource for robust and scalable QA. 

---
# SwiLTra-Bench: The Swiss Legal Translation Benchmark 

**Authors**: Joel Niklaus, Jakob Merane, Luka Nenadic, Sina Ahmadi, Yingqiang Gao, Cyrill A. H. Chevalley, Claude Humbel, Christophe Gösken, Lorenzo Tanzi, Thomas Lüthi, Stefan Palombo, Spencer Poff, Boling Yang, Nan Wu, Matthew Guillod, Robin Mamié, Daniel Brunner, Julio Pereyra, Niko Grupen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01372)  

**Abstract**: In Switzerland legal translation is uniquely important due to the country's four official languages and requirements for multilingual legal documentation. However, this process traditionally relies on professionals who must be both legal experts and skilled translators -- creating bottlenecks and impacting effective access to justice. To address this challenge, we introduce SwiLTra-Bench, a comprehensive multilingual benchmark of over 180K aligned Swiss legal translation pairs comprising laws, headnotes, and press releases across all Swiss languages along with English, designed to evaluate LLM-based translation systems. Our systematic evaluation reveals that frontier models achieve superior translation performance across all document types, while specialized translation systems excel specifically in laws but under-perform in headnotes. Through rigorous testing and human expert validation, we demonstrate that while fine-tuning open SLMs significantly improves their translation quality, they still lag behind the best zero-shot prompted frontier models such as Claude-3.5-Sonnet. Additionally, we present SwiLTra-Judge, a specialized LLM evaluation system that aligns best with human expert assessments. 

---
# SRAG: Structured Retrieval-Augmented Generation for Multi-Entity Question Answering over Wikipedia Graph 

**Authors**: Teng Lin, Yizhang Zhu, Yuyu Luo, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01346)  

**Abstract**: Multi-entity question answering (MEQA) poses significant challenges for large language models (LLMs), which often struggle to consolidate scattered information across multiple documents. An example question might be "What is the distribution of IEEE Fellows among various fields of study?", which requires retrieving information from diverse sources e.g., Wikipedia pages. The effectiveness of current retrieval-augmented generation (RAG) methods is limited by the LLMs' capacity to aggregate insights from numerous pages. To address this gap, this paper introduces a structured RAG (SRAG) framework that systematically organizes extracted entities into relational tables (e.g., tabulating entities with schema columns like "name" and "field of study") and then apply table-based reasoning techniques. Our approach decouples retrieval and reasoning, enabling LLMs to focus on structured data analysis rather than raw text aggregation. Extensive experiments on Wikipedia-based multi-entity QA tasks demonstrate that SRAG significantly outperforms state-of-the-art long-context LLMs and RAG solutions, achieving a 29.6% improvement in accuracy. The results underscore the efficacy of structuring unstructured data to enhance LLMs' reasoning capabilities. 

---
# Same Question, Different Words: A Latent Adversarial Framework for Prompt Robustness 

**Authors**: Tingchen Fu, Fazl Barez  

**Link**: [PDF](https://arxiv.org/pdf/2503.01345)  

**Abstract**: Insensitivity to semantically-preserving variations of prompts (paraphrases) is crucial for reliable behavior and real-world deployment of large language models. However, language models exhibit significant performance degradation when faced with semantically equivalent but differently phrased prompts, and existing solutions either depend on trial-and-error prompt engineering or require computationally expensive inference-time algorithms. In this study, built on the key insight that worst-case prompts exhibit a drift in embedding space, we present Latent Adversarial Paraphrasing (LAP), a dual-loop adversarial framework: the inner loop trains a learnable perturbation to serve as a "latent continuous paraphrase" while preserving semantics through Lagrangian regulation, and the outer loop optimizes the language model parameters on these perturbations. We conduct extensive experiments to demonstrate the effectiveness of LAP across multiple LLM architectures on the RobustAlpaca benchmark with a 0.5%-4% absolution improvement on worst-case win-rate compared with vanilla supervised fine-tuning. 

---
# Answer, Refuse, or Guess? Investigating Risk-Aware Decision Making in Language Models 

**Authors**: Cheng-Kuang Wu, Zhi Rui Tam, Chieh-Yen Lin, Yun-Nung Chen, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.01332)  

**Abstract**: Knowing when to answer or refuse is crucial for safe and reliable decision-making language agents. Although prior work has introduced refusal strategies to boost LMs' reliability, how these models adapt their decisions to different risk levels remains underexplored. We formalize the task of risk-aware decision-making, expose critical weaknesses in existing LMs, and propose skill-decomposition solutions to mitigate them. Our findings show that even cutting-edge LMs--both regular and reasoning models--still require explicit prompt chaining to handle the task effectively, revealing the challenges that must be overcome to achieve truly autonomous decision-making agents. 

---
# WeightedKV: Attention Scores Weighted Key-Value Cache Merging for Large Language Models 

**Authors**: Jian Yuan, Ziwei He, Haoli Bai, Jingwen Leng, Bo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01330)  

**Abstract**: Large Language Models (LLMs) use key-value (KV) cache to reduce redundant computation in autoregressive generation. However, the KV cache size increases linearly during generation, leading to excessive memory usage, especially for long texts. Most KV cache compression methods evict the unimportant KV pairs to maintain a fixed cache size, which leads to the permanent loss of tokens during generation. However, singular value decomposition shows that \textit{values} do not exhibit a strong low-rank property as \textit{keys} do, suggesting that information is distributed more evenly across \textit{values}, in contrast to its more redundant distribution within \textit{keys}. Therefore, methods that evict both \textit{keys} and \textit{values} risk losing crucial information and compromise context integrity, ultimately degrading the output quality. To address this problem, we propose WeightedKV, a novel, training-free approach that discards the \textit{keys} of less important tokens, while merging their \textit{values} into neighboring tokens via a convex combination weighted by their average attention scores. In this way, the retained \textit{keys} serve as anchors that guide the generation process, while the merged \textit{values} provide a rich contextual backdrop. We assess our method on four widely used language modeling datasets, demonstrating superior performance compared to all baseline methods, particularly with a lower budget ratio. 

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
# Causal Tree Extraction from Medical Case Reports: A Novel Task for Experts-like Text Comprehension 

**Authors**: Sakiko Yahata, Zhen Wan, Fei Cheng, Sadao Kurohashi, Hisahiko Sato, Ryozo Nagai  

**Link**: [PDF](https://arxiv.org/pdf/2503.01302)  

**Abstract**: Extracting causal relationships from a medical case report is essential for comprehending the case, particularly its diagnostic process. Since the diagnostic process is regarded as a bottom-up inference, causal relationships in cases naturally form a multi-layered tree structure. The existing tasks, such as medical relation extraction, are insufficient for capturing the causal relationships of an entire case, as they treat all relations equally without considering the hierarchical structure inherent in the diagnostic process. Thus, we propose a novel task, Causal Tree Extraction (CTE), which receives a case report and generates a causal tree with the primary disease as the root, providing an intuitive understanding of a case's diagnostic process. Subsequently, we construct a Japanese case report CTE dataset, J-Casemap, propose a generation-based CTE method that outperforms the baseline by 20.2 points in the human evaluation, and introduce evaluation metrics that reflect clinician preferences. Further experiments also show that J-Casemap enhances the performance of solving other medical tasks, such as question answering. 

---
# Enhancing Non-English Capabilities of English-Centric Large Language Models through Deep Supervision Fine-Tuning 

**Authors**: Wenshuai Huo, Xiaocheng Feng, Yichong Huang, Chengpeng Fu, Baohang Li, Yangfan Ye, Zhirui Zhang, Dandan Tu, Duyu Tang, Yunfei Lu, Hui Wang, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2503.01275)  

**Abstract**: Large language models (LLMs) have demonstrated significant progress in multilingual language understanding and generation. However, due to the imbalance in training data, their capabilities in non-English languages are limited. Recent studies revealed the English-pivot multilingual mechanism of LLMs, where LLMs implicitly convert non-English queries into English ones at the bottom layers and adopt English for thinking at the middle layers. However, due to the absence of explicit supervision for cross-lingual alignment in the intermediate layers of LLMs, the internal representations during these stages may become inaccurate. In this work, we introduce a deep supervision fine-tuning method (DFT) that incorporates additional supervision in the internal layers of the model to guide its workflow. Specifically, we introduce two training objectives on different layers of LLMs: one at the bottom layers to constrain the conversion of the target language into English, and another at the middle layers to constrain reasoning in English. To effectively achieve the guiding purpose, we designed two types of supervision signals: logits and feature, which represent a stricter constraint and a relatively more relaxed guidance. Our method guides the model to not only consider the final generated result when processing non-English inputs but also ensure the accuracy of internal representations. We conducted extensive experiments on typical English-centric large models, LLaMA-2 and Gemma-2, and the results on multiple multilingual datasets show that our method significantly outperforms traditional fine-tuning methods. 

---
# ChatGPT for President! Presupposed content in politicians versus GPT-generated texts 

**Authors**: Davide Garassino, Nicola Brocca, Viviana Masia  

**Link**: [PDF](https://arxiv.org/pdf/2503.01269)  

**Abstract**: This study examines ChatGPT-4's capability to replicate linguistic strategies used in political discourse, focusing on its potential for manipulative language generation. As large language models become increasingly popular for text generation, concerns have grown regarding their role in spreading fake news and propaganda. This research compares real political speeches with those generated by ChatGPT, emphasizing presuppositions (a rhetorical device that subtly influences audiences by packaging some content as already known at the moment of utterance, thus swaying opinions without explicit argumentation). Using a corpus-based pragmatic analysis, this study assesses how well ChatGPT can mimic these persuasive strategies. The findings reveal that although ChatGPT-generated texts contain many manipulative presuppositions, key differences emerge in their frequency, form, and function compared with those of politicians. For instance, ChatGPT often relies on change-of-state verbs used in fixed phrases, whereas politicians use presupposition triggers in more varied and creative ways. Such differences, however, are challenging to detect with the naked eye, underscoring the potential risks posed by large language models in political and public this http URL a corpus-based pragmatic analysis, this study assesses how well ChatGPT can mimic these persuasive strategies. The findings reveal that although ChatGPT-generated texts contain many manipulative presuppositions, key differences emerge in their frequency, form, and function compared with those of politicians. For instance, ChatGPT often relies on change-of-state verbs used in fixed phrases, whereas politicians use presupposition triggers in more varied and creative ways. Such differences, however, are challenging to detect with the naked eye, underscoring the potential risks posed by large language models in political and public discourse. 

---
# Your Model is Overconfident, and Other Lies We Tell Ourselves 

**Authors**: Timothee Mickus, Aman Sinha, Raúl Vázquez  

**Link**: [PDF](https://arxiv.org/pdf/2503.01235)  

**Abstract**: The difficulty intrinsic to a given example, rooted in its inherent ambiguity, is a key yet often overlooked factor in evaluating neural NLP models. We investigate the interplay and divergence among various metrics for assessing intrinsic difficulty, including annotator dissensus, training dynamics, and model confidence. Through a comprehensive analysis using 29 models on three datasets, we reveal that while correlations exist among these metrics, their relationships are neither linear nor monotonic. By disentangling these dimensions of uncertainty, we aim to refine our understanding of data complexity and its implications for evaluating and improving NLP models. 

---
# PEO: Improving Bi-Factorial Preference Alignment with Post-Training Policy Extrapolation 

**Authors**: Yuxuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01233)  

**Abstract**: The alignment of large language models with human values presents a critical challenge, particularly when balancing conflicting objectives like helpfulness and harmlessness. Existing approaches, such as Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO), face notable limitations: RLHF suffers from instability and inefficiency in multi-objective optimization, while DPO lacks mechanisms for dynamic trade-offs. To address these challenges, we propose Post-Training Extrapolation Optimization (PEO), a novel and efficient framework for bi-factorial alignment. PEO generates a family of Pareto-optimal policies in a single training pass by leveraging a three-phase pipeline: (1) aspect-specific learning, (2) generalist initialization via interpolation, and (3) post-training optimization via extrapolation. PEO enables dynamic adaptation to diverse user preferences at inference time without retraining. Our comprehensive experiments across multiple LLMs demonstrate that PEO achieves superior Pareto fronts compared to baselines, offering improved flexibility and computational efficiency. Theoretical analyses further highlight PEO's capacity to overcome optimization bottlenecks, paving the way for scalable, personalized alignment. 

---
# HREB-CRF: Hierarchical Reduced-bias EMA for Chinese Named Entity Recognition 

**Authors**: Sijin Sun, Ming Deng, Xinrui Yu, Liangbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.01217)  

**Abstract**: Incorrect boundary division, complex semantic representation, and differences in pronunciation and meaning often lead to errors in Chinese Named Entity Recognition(CNER). To address these issues, this paper proposes HREB-CRF framework: Hierarchical Reduced-bias EMA with CRF. The proposed method amplifies word boundaries and pools long text gradients through exponentially fixed-bias weighted average of local and global hierarchical attention. Experimental results on the MSRA, Resume, and Weibo datasets show excellent in F1, outperforming the baseline model by 1.1\%, 1.6\%, and 9.8\%. The significant improvement in F1 shows evidences of strong effectiveness and robustness of approach in CNER tasks. 

---
# Cancer Type, Stage and Prognosis Assessment from Pathology Reports using LLMs 

**Authors**: Rachit Saluja, Jacob Rosenthal, Yoav Artzi, David J. Pisapia, Benjamin L. Liechty, Mert R. Sabuncu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01194)  

**Abstract**: Large Language Models (LLMs) have shown significant promise across various natural language processing tasks. However, their application in the field of pathology, particularly for extracting meaningful insights from unstructured medical texts such as pathology reports, remains underexplored and not well quantified. In this project, we leverage state-of-the-art language models, including the GPT family, Mistral models, and the open-source Llama models, to evaluate their performance in comprehensively analyzing pathology reports. Specifically, we assess their performance in cancer type identification, AJCC stage determination, and prognosis assessment, encompassing both information extraction and higher-order reasoning tasks. Based on a detailed analysis of their performance metrics in a zero-shot setting, we developed two instruction-tuned models: Path-llama3.1-8B and Path-GPT-4o-mini-FT. These models demonstrated superior performance in zero-shot cancer type identification, staging, and prognosis assessment compared to the other models evaluated. 

---
# Talking Turns: Benchmarking Audio Foundation Models on Turn-Taking Dynamics 

**Authors**: Siddhant Arora, Zhiyun Lu, Chung-Cheng Chiu, Ruoming Pang, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2503.01174)  

**Abstract**: The recent wave of audio foundation models (FMs) could provide new capabilities for conversational modeling. However, there have been limited efforts to evaluate these audio FMs comprehensively on their ability to have natural and interactive conversations. To engage in meaningful conversation with the end user, we would want the FMs to additionally perform a fluent succession of turns without too much overlapping speech or long stretches of silence. Inspired by this, we ask whether the recently proposed audio FMs can understand, predict, and perform turn-taking events? To answer this, we propose a novel evaluation protocol that can assess spoken dialog system's turn-taking capabilities using a supervised model as a judge that has been trained to predict turn-taking events in human-human conversations. Using this protocol, we present the first comprehensive user study that evaluates existing spoken dialogue systems on their ability to perform turn-taking events and reveal many interesting insights, such as they sometimes do not understand when to speak up, can interrupt too aggressively and rarely backchannel. We further evaluate multiple open-source and proprietary audio FMs accessible through APIs on carefully curated test benchmarks from Switchboard to measure their ability to understand and predict turn-taking events and identify significant room for improvement. We will open source our evaluation platform to promote the development of advanced conversational AI systems. 

---
# Large Language Models for Healthcare Text Classification: A Systematic Review 

**Authors**: Hajar Sakai, Sarah S. Lam  

**Link**: [PDF](https://arxiv.org/pdf/2503.01159)  

**Abstract**: Large Language Models (LLMs) have fundamentally transformed approaches to Natural Language Processing (NLP) tasks across diverse domains. In healthcare, accurate and cost-efficient text classification is crucial, whether for clinical notes analysis, diagnosis coding, or any other task, and LLMs present promising potential. Text classification has always faced multiple challenges, including manual annotation for training, handling imbalanced data, and developing scalable approaches. With healthcare, additional challenges are added, particularly the critical need to preserve patients' data privacy and the complexity of the medical terminology. Numerous studies have been conducted to leverage LLMs for automated healthcare text classification and contrast the results with existing machine learning-based methods where embedding, annotation, and training are traditionally required. Existing systematic reviews about LLMs either do not specialize in text classification or do not focus on the healthcare domain. This research synthesizes and critically evaluates the current evidence found in the literature regarding the use of LLMs for text classification in a healthcare setting. Major databases (e.g., Google Scholar, Scopus, PubMed, Science Direct) and other resources were queried, which focused on the papers published between 2018 and 2024 within the framework of PRISMA guidelines, which resulted in 65 eligible research articles. These were categorized by text classification type (e.g., binary classification, multi-label classification), application (e.g., clinical decision support, public health and opinion analysis), methodology, type of healthcare text, and metrics used for evaluation and validation. This review reveals the existing gaps in the literature and suggests future research lines that can be investigated and explored. 

---
# Nature-Inspired Population-Based Evolution of Large Language Models 

**Authors**: Yiqun Zhang, Peng Ye, Xiaocui Yang, Shi Feng, Shufei Zhang, Lei Bai, Wanli Ouyang, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01155)  

**Abstract**: Evolution, the engine behind the survival and growth of life on Earth, operates through the population-based process of reproduction. Inspired by this principle, this paper formally defines a newly emerging problem -- the population-based evolution of large language models (LLMs) -- and introduces a novel framework. Starting with a population of parent LLMs, our framework enables the population to evolve through four key operations: (i) crossover, merging the weights of different parents to create offspring LLMs, (ii) mutation, introducing small, random changes to model weights to foster diversity, (iii) selection, prioritizing high-performing models, and (iv) succession, transferring the learned experience from parent to offspring LLMs. With only 200 samples per new task, the LLM population evolves rapidly to adapt to the task at hand, without any gradients. Experiments on 12 datasets show that our framework consistently outperforms existing multi-LLM merging and adaptation methods, achieving accuracy gains of up to 54.8% over the best LLM in the initial population. Moreover, our framework allows for the evolution of LLMs across multiple new tasks simultaneously, scaling effectively with populations of up to 40 LLMs, and even zero-shot generalization to unseen held-out tasks. We have open-sourced the code on GitHub and released the weights of 10 parent LLMs, fine-tuned from gemma-2-2b-it, on HuggingFace$, enabling reproduction of our proposed framework using just a single 4090 GPU with 24GB memory, without any performance degradation. 

---
# ReaderLM-v2: Small Language Model for HTML to Markdown and JSON 

**Authors**: Feng Wang, Zesheng Shi, Bo Wang, Nan Wang, Han Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.01151)  

**Abstract**: We present ReaderLM-v2, a compact 1.5 billion parameter language model designed for efficient web content extraction. Our model processes documents up to 512K tokens, transforming messy HTML into clean Markdown or JSON formats with high accuracy -- making it an ideal tool for grounding large language models. The model's effectiveness results from two key innovations: (1) a three-stage data synthesis pipeline that generates high quality, diverse training data by iteratively drafting, refining, and critiquing web content extraction; and (2) a unified training framework combining continuous pre-training with multi-objective optimization. Intensive evaluation demonstrates that ReaderLM-v2 outperforms GPT-4o-2024-08-06 and other larger models by 15-20\% on carefully curated benchmarks, particularly excelling at documents exceeding 100K tokens, while maintaining significantly lower computational requirements. 

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
# Beyond QA Pairs: Assessing Parameter-Efficient Fine-Tuning for Fact Embedding in LLMs 

**Authors**: Shivam Ratnakar, Abhiroop Talasila, Raghav Chamadiya, Nikhil Agarwal, Vinayak K Doifode  

**Link**: [PDF](https://arxiv.org/pdf/2503.01131)  

**Abstract**: This paper presents an extensive examination of Parameter-Efficient Fine-Tuning (PEFT) for embedding domain specific facts into Large Language Models (LLMs), focusing on improving the fine-tuning process by categorizing question-answer (QA) pairs into Factual and Conceptual classes using a BERT-based classifier. Two distinct Llama-2 models are fine-tuned based on these classifications and evaluated using larger models like GPT-3.5 Turbo and Gemini. Our results indicate that models trained on conceptual datasets outperform those trained on factual datasets. Additionally, we compare the efficiency of two synthetic fine-tuning dataset generation techniques, D-RAG and D-Naive, with D-Naive demonstrating superior performance. Although PEFT has shown effectiveness, our research indicates that it may not be the most optimal method for embedding facts into LLMs. However, it has demonstrated exceptional performance in instruction-based tasks. Our findings are reinforced by a 1000-sample dataset in the data center domain, where the fine-tuned Llama-2 7B model significantly outperforms the baseline model in generating product recommendations. Our study highlights the importance of QA pair categorization and synthetic dataset generation techniques in enhancing the performance of LLMs in specific domains. 

---
# Precise Localization of Memories: A Fine-grained Neuron-level Knowledge Editing Technique for LLMs 

**Authors**: Haowen Pan, Xiaozhi Wang, Yixin Cao, Zenglin Shi, Xun Yang, Juanzi Li, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01090)  

**Abstract**: Knowledge editing aims to update outdated information in Large Language Models (LLMs). A representative line of study is locate-then-edit methods, which typically employ causal tracing to identify the modules responsible for recalling factual knowledge about entities. However, we find these methods are often sensitive only to changes in the subject entity, leaving them less effective at adapting to changes in relations. This limitation results in poor editing locality, which can lead to the persistence of irrelevant or inaccurate facts, ultimately compromising the reliability of LLMs. We believe this issue arises from the insufficient precision of knowledge localization. To address this, we propose a Fine-grained Neuron-level Knowledge Editing (FiNE) method that enhances editing locality without affecting overall success rates. By precisely identifying and modifying specific neurons within feed-forward networks, FiNE significantly improves knowledge localization and editing. Quantitative experiments demonstrate that FiNE efficiently achieves better overall performance compared to existing techniques, providing new insights into the localization and modification of knowledge within LLMs. 

---
# Efficient or Powerful? Trade-offs Between Machine Learning and Deep Learning for Mental Illness Detection on Social Media 

**Authors**: Zhanyi Ding, Zhongyan Wang, Yeyubei Zhang, Yuchen Cao, Yunchong Liu, Xiaorui Shen, Yexin Tian, Jianglai Dai  

**Link**: [PDF](https://arxiv.org/pdf/2503.01082)  

**Abstract**: Social media platforms provide valuable insights into mental health trends by capturing user-generated discussions on conditions such as depression, anxiety, and suicidal ideation. Machine learning (ML) and deep learning (DL) models have been increasingly applied to classify mental health conditions from textual data, but selecting the most effective model involves trade-offs in accuracy, interpretability, and computational efficiency. This study evaluates multiple ML models, including logistic regression, random forest, and LightGBM, alongside deep learning architectures such as ALBERT and Gated Recurrent Units (GRUs), for both binary and multi-class classification of mental health conditions. Our findings indicate that ML and DL models achieve comparable classification performance on medium-sized datasets, with ML models offering greater interpretability through variable importance scores, while DL models are more robust to complex linguistic patterns. Additionally, ML models require explicit feature engineering, whereas DL models learn hierarchical representations directly from text. Logistic regression provides the advantage of capturing both positive and negative associations between features and mental health conditions, whereas tree-based models prioritize decision-making power through split-based feature selection. This study offers empirical insights into the advantages and limitations of different modeling approaches and provides recommendations for selecting appropriate methods based on dataset size, interpretability needs, and computational constraints. 

---
# Scientific Reasoning: Assessment of Multimodal Generative LLMs 

**Authors**: Florian Dreyer, Ekaterina Kolos, Daria Matiash  

**Link**: [PDF](https://arxiv.org/pdf/2503.01064)  

**Abstract**: Large language models (LLMs) can answer questions and reason about complex tasks, also from the scientific domain. We assess several multimodal LLMs (MLLMs) on ScienceQA and find that Gemini models show the highest accuracy with little context, and the highest textual similarity to human explanations with richer context. Adapter-tuning of smaller MLLMs did not lead to any reliable performance. Training from Gemini outputs consistently underperformed training from the original data. 

---
# AI-Invented Tonal Languages: Preventing a Machine Lingua Franca Beyond Human Understanding 

**Authors**: David Noever  

**Link**: [PDF](https://arxiv.org/pdf/2503.01063)  

**Abstract**: This paper investigates the potential for large language models (LLMs) to develop private tonal languages for machine-to-machine (M2M) communication. Inspired by cryptophasia in human twins (affecting up to 50% of twin births) and natural tonal languages like Mandarin and Vietnamese, we implement a precise character-to-frequency mapping system that encodes the full ASCII character set (32-126) using musical semitones. Each character is assigned a unique frequency, creating a logarithmic progression beginning with space (220 Hz) and ending with tilde (50,175.42 Hz). This spans approximately 7.9 octaves, with higher characters deliberately mapped to ultrasonic frequencies beyond human perception (>20 kHz). Our implemented software prototype demonstrates this encoding through visualization, auditory playback, and ABC musical notation, allowing for analysis of information density and transmission speed. Testing reveals that tonal encoding can achieve information rates exceeding human speech while operating partially outside human perceptual boundaries. This work responds directly to concerns about AI systems catastrophically developing private languages within the next five years, providing a concrete prototype software example of how such communication might function and the technical foundation required for its emergence, detection, and governance. 

---
# Language-agnostic, automated assessment of listeners' speech recall using large language models 

**Authors**: Björn Herrmann  

**Link**: [PDF](https://arxiv.org/pdf/2503.01045)  

**Abstract**: Speech-comprehension difficulties are common among older people. Standard speech tests do not fully capture such difficulties because the tests poorly resemble the context-rich, story-like nature of ongoing conversation and are typically available only in a country's dominant/official language (e.g., English), leading to inaccurate scores for native speakers of other languages. Assessments for naturalistic, story speech in multiple languages require accurate, time-efficient scoring. The current research leverages modern large language models (LLMs) in native English speakers and native speakers of 10 other languages to automate the generation of high-quality, spoken stories and scoring of speech recall in different languages. Participants listened to and freely recalled short stories (in quiet/clear and in babble noise) in their native language. LLM text-embeddings and LLM prompt engineering with semantic similarity analyses to score speech recall revealed sensitivity to known effects of temporal order, primacy/recency, and background noise, and high similarity of recall scores across languages. The work overcomes limitations associated with simple speech materials and testing of closed native-speaker groups because recall data of varying length and details can be mapped across languages with high accuracy. The full automation of speech generation and recall scoring provides an important step towards comprehension assessments of naturalistic speech with clinical applicability. 

---
# Language Models Predict Empathy Gaps Between Social In-groups and Out-groups 

**Authors**: Yu Hou, Hal Daumé III, Rachel Rudinger  

**Link**: [PDF](https://arxiv.org/pdf/2503.01030)  

**Abstract**: Studies of human psychology have demonstrated that people are more motivated to extend empathy to in-group members than out-group members (Cikara et al., 2011). In this study, we investigate how this aspect of intergroup relations in humans is replicated by LLMs in an emotion intensity prediction task. In this task, the LLM is given a short description of an experience a person had that caused them to feel a particular emotion; the LLM is then prompted to predict the intensity of the emotion the person experienced on a numerical scale. By manipulating the group identities assigned to the LLM's persona (the "perceiver") and the person in the narrative (the "experiencer"), we measure how predicted emotion intensities differ between in-group and out-group settings. We observe that LLMs assign higher emotion intensity scores to in-group members than out-group members. This pattern holds across all three types of social groupings we tested: race/ethnicity, nationality, and religion. We perform an in-depth analysis on Llama-3.1-8B, the model which exhibited strongest intergroup bias among those tested. 

---
# Evaluating Polish linguistic and cultural competency in large language models 

**Authors**: Sławomir Dadas, Małgorzata Grębowiec, Michał Perełkiewicz, Rafał Poświata  

**Link**: [PDF](https://arxiv.org/pdf/2503.00995)  

**Abstract**: Large language models (LLMs) are becoming increasingly proficient in processing and generating multilingual texts, which allows them to address real-world problems more effectively. However, language understanding is a far more complex issue that goes beyond simple text analysis. It requires familiarity with cultural context, including references to everyday life, historical events, traditions, folklore, literature, and pop culture. A lack of such knowledge can lead to misinterpretations and subtle, hard-to-detect errors. To examine language models' knowledge of the Polish cultural context, we introduce the Polish linguistic and cultural competency benchmark, consisting of 600 manually crafted questions. The benchmark is divided into six categories: history, geography, culture & tradition, art & entertainment, grammar, and vocabulary. As part of our study, we conduct an extensive evaluation involving over 30 open-weight and commercial LLMs. Our experiments provide a new perspective on Polish competencies in language models, moving past traditional natural language processing tasks and general knowledge assessment. 

---
# Enhancing Text Editing for Grammatical Error Correction: Arabic as a Case Study 

**Authors**: Bashar Alhafni, Nizar Habash  

**Link**: [PDF](https://arxiv.org/pdf/2503.00985)  

**Abstract**: Text editing frames grammatical error correction (GEC) as a sequence tagging problem, where edit tags are assigned to input tokens, and applying these edits results in the corrected text. This approach has gained attention for its efficiency and interpretability. However, while extensively explored for English, text editing remains largely underexplored for morphologically rich languages like Arabic. In this paper, we introduce a text editing approach that derives edit tags directly from data, eliminating the need for language-specific edits. We demonstrate its effectiveness on Arabic, a diglossic and morphologically rich language, and investigate the impact of different edit representations on model performance. Our approach achieves SOTA results on two Arabic GEC benchmarks and performs on par with SOTA on two others. Additionally, our models are over six times faster than existing Arabic GEC systems, making our approach more practical for real-world applications. Finally, we explore ensemble models, demonstrating how combining different models leads to further performance improvements. We make our code, data, and pretrained models publicly available. 

---
# Dialogue Without Limits: Constant-Sized KV Caches for Extended Responses in LLMs 

**Authors**: Ravi Ghadia, Avinash Kumar, Gaurav Jain, Prashant Nair, Poulami Das  

**Link**: [PDF](https://arxiv.org/pdf/2503.00979)  

**Abstract**: Autoregressive Transformers rely on Key-Value (KV) caching to accelerate inference. However, the linear growth of the KV cache with context length leads to excessive memory consumption and bandwidth constraints. This bottleneck is particularly problematic in real-time applications -- such as chatbots and interactive assistants -- where low latency and high memory efficiency are critical. Existing methods drop distant tokens or compress states in a lossy manner, sacrificing accuracy by discarding vital context or introducing bias.
We propose MorphKV, an inference-time technique that maintains a constant-sized KV cache while preserving accuracy. MorphKV balances long-range dependencies and local coherence during text generation. It eliminates early-token bias while retaining high-fidelity context by adaptively ranking tokens through correlation-aware selection. Unlike heuristic retention or lossy compression, MorphKV iteratively refines the KV cache via lightweight updates guided by attention patterns of recent tokens. This approach captures inter-token correlation with greater accuracy, crucial for tasks like content creation and code generation. Our studies on long-response tasks show 52.9$\%$ memory savings and 18.2$\%$ higher accuracy on average compared to state-of-the-art prior works, enabling efficient real-world deployment. 

---
# Layered Insights: Generalizable Analysis of Authorial Style by Leveraging All Transformer Layers 

**Authors**: Milad Alshomary, Nikhil Reddy Varimalla, Vishal Anand, Kathleen McKeown  

**Link**: [PDF](https://arxiv.org/pdf/2503.00958)  

**Abstract**: We propose a new approach for the authorship attribution task that leverages the various linguistic representations learned at different layers of pre-trained transformer-based models. We evaluate our approach on three datasets, comparing it to a state-of-the-art baseline in in-domain and out-of-domain scenarios. We found that utilizing various transformer layers improves the robustness of authorship attribution models when tested on out-of-domain data, resulting in new state-of-the-art results. Our analysis gives further insights into how our model's different layers get specialized in representing certain stylistic features that benefit the model when tested out of the domain. 

---
# SemViQA: A Semantic Question Answering System for Vietnamese Information Fact-Checking 

**Authors**: Nam V. Nguyen, Dien X. Tran, Thanh T. Tran, Anh T. Hoang, Tai V. Duong, Di T. Le, Phuc-Lu Le  

**Link**: [PDF](https://arxiv.org/pdf/2503.00955)  

**Abstract**: The rise of misinformation, exacerbated by Large Language Models (LLMs) like GPT and Gemini, demands robust fact-checking solutions, especially for low-resource languages like Vietnamese. Existing methods struggle with semantic ambiguity, homonyms, and complex linguistic structures, often trading accuracy for efficiency. We introduce SemViQA, a novel Vietnamese fact-checking framework integrating Semantic-based Evidence Retrieval (SER) and Two-step Verdict Classification (TVC). Our approach balances precision and speed, achieving state-of-the-art results with 78.97\% strict accuracy on ISE-DSC01 and 80.82\% on ViWikiFC, securing 1st place in the UIT Data Science Challenge. Additionally, SemViQA Faster improves inference speed 7x while maintaining competitive accuracy. SemViQA sets a new benchmark for Vietnamese fact verification, advancing the fight against misinformation. The source code is available at: this https URL. 

---
# HiBench: Benchmarking LLMs Capability on Hierarchical Structure Reasoning 

**Authors**: Zhuohang Jiang, Pangjing Wu, Ziran Liang, Peter Q. Chen, Xu Yuan, Ye Jia, Jiancheng Tu, Chen Li, Peter H.F. Ng, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00912)  

**Abstract**: Structure reasoning is a fundamental capability of large language models (LLMs), enabling them to reason about structured commonsense and answer multi-hop questions. However, existing benchmarks for structure reasoning mainly focus on horizontal and coordinate structures (\emph{e.g.} graphs), overlooking the hierarchical relationships within them. Hierarchical structure reasoning is crucial for human cognition, particularly in memory organization and problem-solving. It also plays a key role in various real-world tasks, such as information extraction and decision-making. To address this gap, we propose HiBench, the first framework spanning from initial structure generation to final proficiency assessment, designed to benchmark the hierarchical reasoning capabilities of LLMs systematically. HiBench encompasses six representative scenarios, covering both fundamental and practical aspects, and consists of 30 tasks with varying hierarchical complexity, totaling 39,519 queries. To evaluate LLMs comprehensively, we develop five capability dimensions that depict different facets of hierarchical structure understanding. Through extensive evaluation of 20 LLMs from 10 model families, we reveal key insights into their capabilities and limitations: 1) existing LLMs show proficiency in basic hierarchical reasoning tasks; 2) they still struggle with more complex structures and implicit hierarchical representations, especially in structural modification and textual reasoning. Based on these findings, we create a small yet well-designed instruction dataset, which enhances LLMs' performance on HiBench by an average of 88.84\% (Llama-3.1-8B) and 31.38\% (Qwen2.5-7B) across all tasks. The HiBench dataset and toolkit are available here, this https URL, to encourage evaluation. 

---
# Unveiling Biases while Embracing Sustainability: Assessing the Dual Challenges of Automatic Speech Recognition Systems 

**Authors**: Ajinkya Kulkarni, Atharva Kulkarni, Miguel Couceiro, Isabel Trancoso  

**Link**: [PDF](https://arxiv.org/pdf/2503.00907)  

**Abstract**: In this paper, we present a bias and sustainability focused investigation of Automatic Speech Recognition (ASR) systems, namely Whisper and Massively Multilingual Speech (MMS), which have achieved state-of-the-art (SOTA) performances. Despite their improved performance in controlled settings, there remains a critical gap in understanding their efficacy and equity in real-world scenarios. We analyze ASR biases w.r.t. gender, accent, and age group, as well as their effect on downstream tasks. In addition, we examine the environmental impact of ASR systems, scrutinizing the use of large acoustic models on carbon emission and energy consumption. We also provide insights into our empirical analyses, offering a valuable contribution to the claims surrounding bias and sustainability in ASR systems. 

---
# Instruct-of-Reflection: Enhancing Large Language Models Iterative Reflection Capabilities via Dynamic-Meta Instruction 

**Authors**: Liping Liu, Chunhong Zhang, Likang Wu, Chuang Zhao, Zheng Hu, Ming He, Jianping Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.00902)  

**Abstract**: Self-reflection for Large Language Models (LLMs) has gained significant attention. Existing approaches involve models iterating and improving their previous responses based on LLMs' internal reflection ability or external feedback. However, recent research has raised doubts about whether intrinsic self-correction without external feedback may even degrade performance. Based on our empirical evidence, we find that current static reflection methods may lead to redundant, drift, and stubborn issues. To mitigate this, we introduce Instruct-of-Reflection (IoRT), a novel and general reflection framework that leverages dynamic-meta instruction to enhance the iterative reflection capability of LLMs. Specifically, we propose the instructor driven by the meta-thoughts and self-consistency classifier, generates various instructions, including refresh, stop, and select, to guide the next reflection iteration. Our experiments demonstrate that IoRT achieves an average improvement of 10.1% over established baselines in mathematical and commonsense reasoning tasks, highlighting its efficacy and applicability. 

---
# DUAL: Diversity and Uncertainty Active Learning for Text Summarization 

**Authors**: Petros Stylianos Giouroukis, Alexios Gidiotis, Grigorios Tsoumakas  

**Link**: [PDF](https://arxiv.org/pdf/2503.00867)  

**Abstract**: With the rise of large language models, neural text summarization has advanced significantly in recent years. However, even state-of-the-art models continue to rely heavily on high-quality human-annotated data for training and evaluation. Active learning is frequently used as an effective way to collect such datasets, especially when annotation resources are scarce. Active learning methods typically prioritize either uncertainty or diversity but have shown limited effectiveness in summarization, often being outperformed by random sampling. We present Diversity and Uncertainty Active Learning (DUAL), a novel algorithm that combines uncertainty and diversity to iteratively select and annotate samples that are both representative of the data distribution and challenging for the current model. DUAL addresses the selection of noisy samples in uncertainty-based methods and the limited exploration scope of diversity-based methods. Through extensive experiments with different summarization models and benchmark datasets, we demonstrate that DUAL consistently matches or outperforms the best performing strategies. Using visualizations and quantitative metrics, we provide valuable insights into the effectiveness and robustness of different active learning strategies, in an attempt to understand why these strategies haven't performed consistently in text summarization. Finally, we show that DUAL strikes a good balance between diversity and robustness. 

---
# Babel: Open Multilingual Large Language Models Serving Over 90% of Global Speakers 

**Authors**: Yiran Zhao, Chaoqun Liu, Yue Deng, Jiahao Ying, Mahani Aljunied, Zhaodonghui Li, Lidong Bing, Hou Pong Chan, Yu Rong, Deli Zhao, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00865)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing (NLP), yet open-source multilingual LLMs remain scarce, with existing models often limited in language coverage. Such models typically prioritize well-resourced languages, while widely spoken but under-resourced languages are often overlooked. To address this disparity, we introduce $\texttt{Babel}$, an open multilingual LLM that covers the top 25 languages by number of speakers, supports over 90% of the global population, and includes many languages neglected by other open multilingual LLMs. Unlike traditional continue pretraining approaches, Babel expands its parameter count through a layer extension technique that elevates Babel's performance ceiling. We introduce two variants: $\texttt{Babel-9B}$, designed for efficient inference and fine-tuning, and $\texttt{Babel-83B}$, which sets a new standard for open multilingual LLMs. Extensive evaluations on multilingual tasks demonstrate its superior performance compared to open LLMs of comparable size. In addition, using open-source supervised fine-tuning datasets, Babel achieves remarkable performance, with Babel-9B-Chat leading among 10B-sized LLMs and Babel-83B-Chat setting a new standard for multilingual tasks, reaching the same level of commercial models. 

---
# Argument Summarization and its Evaluation in the Era of Large Language Models 

**Authors**: Moritz Altemeyer, Steffen Eger, Johannes Daxenberger, Tim Altendorf, Philipp Cimiano, Benjamin Schiller  

**Link**: [PDF](https://arxiv.org/pdf/2503.00847)  

**Abstract**: Large Language Models (LLMs) have revolutionized various Natural Language Generation (NLG) tasks, including Argument Summarization (ArgSum), a key subfield of Argument Mining (AM). This paper investigates the integration of state-of-the-art LLMs into ArgSum, including for its evaluation. In particular, we propose a novel prompt-based evaluation scheme, and validate it through a novel human benchmark dataset. Our work makes three main contributions: (i) the integration of LLMs into existing ArgSum frameworks, (ii) the development of a new LLM-based ArgSum system, benchmarked against prior methods, and (iii) the introduction of an advanced LLM-based evaluation scheme. We demonstrate that the use of LLMs substantially improves both the generation and evaluation of argument summaries, achieving state-of-the-art results and advancing the field of ArgSum. 

---
# Rewarding Graph Reasoning Process makes LLMs more Generalized Reasoners 

**Authors**: Miao Peng, Nuo Chen, Zongrui Suo, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00845)  

**Abstract**: Despite significant advancements in Large Language Models (LLMs), developing advanced reasoning capabilities in LLMs remains a key challenge. Process Reward Models (PRMs) have demonstrated exceptional promise in enhancing reasoning by providing step-wise feedback, particularly in the context of mathematical reasoning. However, their application to broader reasoning domains remains understudied, largely due to the high costs associated with manually creating step-level supervision. In this work, we explore the potential of PRMs in graph reasoning problems - a domain that demands sophisticated multi-step reasoning and offers opportunities for automated step-level data generation using established graph algorithms. We introduce GraphSILO, the largest dataset for graph reasoning problems with fine-grained step-wise labels, built using automated Task-oriented Trajectories and Monte Carlo Tree Search (MCTS) to generate detailed reasoning steps with step-wise labels. Building upon this dataset, we train GraphPRM, the first PRM designed for graph reasoning problems, and evaluate its effectiveness in two key settings: inference-time scaling and reinforcement learning via Direct Preference Optimization (DPO). Experimental results show that GraphPRM significantly improves LLM performance across 13 graph reasoning tasks, delivering a 9% gain for Qwen2.5-7B and demonstrating transferability to new graph reasoning datasets and new reasoning domains like mathematical problem-solving. Notably, GraphPRM enhances LLM performance on GSM8K and Math500, underscoring the cross-domain applicability of graph-based reasoning rewards. Our findings highlight the potential of PRMs in advancing reasoning across diverse domains, paving the way for more versatile and effective LLMs. 

---
# Waste Not, Want Not; Recycled Gumbel Noise Improves Consistency in Natural Language Generation 

**Authors**: Damien de Mijolla, Hannan Saddiq, Kim Moore  

**Link**: [PDF](https://arxiv.org/pdf/2503.00831)  

**Abstract**: Consistency in the output of language models is critical for their reliability and practical utility. Due to their training objective, language models learn to model the full space of possible continuations, leading to outputs that can vary significantly in style and content, even for similar or repeated inputs. To address this, we propose a novel decoding algorithm that enhances response consistency across different prompts with no degradation in response quality. By incorporating a latent variable into the next-token sampling process based on the Gumbel reparametrisation trick, our method outperforms standard sampling by up to 10% across semantic and stylistic consistency benchmarks. Additionally, our approach integrates seamlessly with existing sampling methods with negligible computational overhead, providing a practical solution for improving the reliability of language model outputs. 

---
# Predictive Data Selection: The Data That Predicts Is the Data That Teaches 

**Authors**: Kashun Shum, Yuzhen Huang, Hongjian Zou, Ding Qi, Yixuan Liao, Xiaoxin Chen, Qian Liu, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2503.00808)  

**Abstract**: Language model pretraining involves training on extensive corpora, where data quality plays a pivotal role. In this work, we aim to directly estimate the contribution of data during pretraining and select pretraining data in an efficient manner. Specifically, we draw inspiration from recent findings showing that compression efficiency (i.e., the normalized loss) of diverse models on certain text correlates strongly with their downstream performance, when the text domain aligns with the downstream benchmark (Huang et al., 2024). Building on this observation, we hypothesize that data on which model losses are predictive of downstream abilities also contribute effectively to learning. To leverage this insight, we introduce data selection based on data's Predictive strength (Preselect), a lightweight and efficient data selection method that requires training and deploying only a fastText-based scorer. Through comprehensive experiments with 1B and 3B parameter models, we demonstrate that models trained on 30B tokens selected with PreSelect surpasses the performance of a vanilla baseline trained on 300B tokens, achieving a 10x reduction in compute requirements. Furthermore, PreSelect significantly outperforms other competitive data selection baselines, such as DCLM and FineWeb-Edu on a scale of 3B models trained on 100B tokens. We open-source our trained data selection scorer along with the curated datasets at this https URL. 

---
# DuoDecoding: Hardware-aware Heterogeneous Speculative Decoding with Dynamic Multi-Sequence Drafting 

**Authors**: Kai Lv, Honglin Guo, Qipeng Guo, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00784)  

**Abstract**: Large language models (LLMs) exhibit exceptional performance across a wide range of tasks; however, their token-by-token autoregressive generation process significantly hinders inference speed. Speculative decoding presents a promising draft-then-verify framework that reduces generation latency while maintaining output distribution fidelity. Nevertheless, the draft model introduces additional computational overhead, becoming a performance bottleneck and increasing the time to first token (TTFT). Previous approaches to mitigate draft model overhead have primarily relied on heuristics and generally failed to match the quality of the draft language models. To address these challenges, we propose DuoDecoding, a novel approach that strategically deploys the draft and target models on the CPU and GPU respectively, enabling parallel decoding while preserving draft quality. Our method incorporates a hardware-aware optimal draft budget to minimize idle times and employs dynamic multi-sequence drafting to enhance draft quality. Extensive experiments across seven tasks show that DuoDecoding achieves up to 2.61x speedup in generation latency, while reducing TTFT to 83% of that in conventional speculative decoding. The Code is available at this https URL. 

---
# Evaluating Personalized Tool-Augmented LLMs from the Perspectives of Personalization and Proactivity 

**Authors**: Yupu Hao, Pengfei Cao, Zhuoran Jin, Huanxuan Liao, Yubo Chen, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.00771)  

**Abstract**: Personalized tool utilization is essential for aligning large language models (LLMs) with user preference in interaction scenarios with various tools. However, most of the current benchmarks primarily focus on either personalization of text generation or direct tool-utilizing, without considering both. In this work, we introduce a novel benchmark ETAPP for evaluating personalized tool invocation, establishing a sandbox environment, and a comprehensive dataset of 800 testing cases covering diverse user profiles. To improve the accuracy of our evaluation, we propose a key-point-based LLM evaluation method, mitigating biases in the LLM-as-a-judge system by manually annotating key points for each test case and providing them to LLM as the reference. Additionally, we evaluate the excellent LLMs and provide an in-depth analysis. Furthermore, we investigate the impact of different tool-invoking strategies on LLMs' personalization performance and the effects of fine-tuning in our task. The effectiveness of our preference-setting and key-point-based evaluation method is also validated. Our findings offer insights into improving personalized LLM agents. Our Code is available at this https URL. 

---
# RAPID: Efficient Retrieval-Augmented Long Text Generation with Writing Planning and Information Discovery 

**Authors**: Hongchao Gu, Dexun Li, Kuicai Dong, Hao Zhang, Hang Lv, Hao Wang, Defu Lian, Yong Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.00751)  

**Abstract**: Generating knowledge-intensive and comprehensive long texts, such as encyclopedia articles, remains significant challenges for Large Language Models. It requires not only the precise integration of facts but also the maintenance of thematic coherence throughout the article. Existing methods, such as direct generation and multi-agent discussion, often struggle with issues like hallucinations, topic incoherence, and significant latency. To address these challenges, we propose RAPID, an efficient retrieval-augmented long text generation framework. RAPID consists of three main modules: (1) Retrieval-augmented preliminary outline generation to reduce hallucinations, (2) Attribute-constrained search for efficient information discovery, (3) Plan-guided article generation for enhanced coherence. Extensive experiments on our newly compiled benchmark dataset, FreshWiki-2024, demonstrate that RAPID significantly outperforms state-of-the-art methods across a wide range of evaluation metrics (e.g. long-text generation, outline quality, latency, etc). Our work provides a robust and efficient solution to the challenges of automated long-text generation. 

---
# Unmasking Digital Falsehoods: A Comparative Analysis of LLM-Based Misinformation Detection Strategies 

**Authors**: Tianyi Huang, Jingyuan Yi, Peiyang Yu, Xiaochuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00724)  

**Abstract**: The proliferation of misinformation on social media has raised significant societal concerns, necessitating robust detection mechanisms. Large Language Models such as GPT-4 and LLaMA2 have been envisioned as possible tools for detecting misinformation based on their advanced natural language understanding and reasoning capabilities. This paper conducts a comparison of LLM-based approaches to detecting misinformation between text-based, multimodal, and agentic approaches. We evaluate the effectiveness of fine-tuned models, zero-shot learning, and systematic fact-checking mechanisms in detecting misinformation across different topic domains like public health, politics, and finance. We also discuss scalability, generalizability, and explainability of the models and recognize key challenges such as hallucination, adversarial attacks on misinformation, and computational resources. Our findings point towards the importance of hybrid approaches that pair structured verification protocols with adaptive learning techniques to enhance detection accuracy and explainability. The paper closes by suggesting potential avenues of future work, including real-time tracking of misinformation, federated learning, and cross-platform detection models. 

---
# An evaluation of DeepSeek Models in Biomedical Natural Language Processing 

**Authors**: Zaifu Zhan, Shuang Zhou, Huixue Zhou, Jiawen Deng, Yu Hou, Jeremy Yeung, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00624)  

**Abstract**: The advancement of Large Language Models (LLMs) has significantly impacted biomedical Natural Language Processing (NLP), enhancing tasks such as named entity recognition, relation extraction, event extraction, and text classification. In this context, the DeepSeek series of models have shown promising potential in general NLP tasks, yet their capabilities in the biomedical domain remain underexplored. This study evaluates multiple DeepSeek models (Distilled-DeepSeek-R1 series and Deepseek-LLMs) across four key biomedical NLP tasks using 12 datasets, benchmarking them against state-of-the-art alternatives (Llama3-8B, Qwen2.5-7B, Mistral-7B, Phi-4-14B, Gemma-2-9B). Our results reveal that while DeepSeek models perform competitively in named entity recognition and text classification, challenges persist in event and relation extraction due to precision-recall trade-offs. We provide task-specific model recommendations and highlight future research directions. This evaluation underscores the strengths and limitations of DeepSeek models in biomedical NLP, guiding their future deployment and optimization. 

---
# Zero-Shot Keyphrase Generation: Investigating Specialized Instructions and Multi-Sample Aggregation on Large Language Models 

**Authors**: Jayanth Mohan, Jishnu Ray Chowdhury, Tomas Malik, Cornelia Caragea  

**Link**: [PDF](https://arxiv.org/pdf/2503.00597)  

**Abstract**: Keyphrases are the essential topical phrases that summarize a document. Keyphrase generation is a long-standing NLP task for automatically generating keyphrases for a given document. While the task has been comprehensively explored in the past via various models, only a few works perform some preliminary analysis of Large Language Models (LLMs) for the task. Given the impact of LLMs in the field of NLP, it is important to conduct a more thorough examination of their potential for keyphrase generation. In this paper, we attempt to meet this demand with our research agenda. Specifically, we focus on the zero-shot capabilities of open-source instruction-tuned LLMs (Phi-3, Llama-3) and the closed-source GPT-4o for this task. We systematically investigate the effect of providing task-relevant specialized instructions in the prompt. Moreover, we design task-specific counterparts to self-consistency-style strategies for LLMs and show significant benefits from our proposals over the baselines. 

---
# BadJudge: Backdoor Vulnerabilities of LLM-as-a-Judge 

**Authors**: Terry Tong, Fei Wang, Zhe Zhao, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.00596)  

**Abstract**: This paper proposes a novel backdoor threat attacking the LLM-as-a-Judge evaluation regime, where the adversary controls both the candidate and evaluator model. The backdoored evaluator victimizes benign users by unfairly assigning inflated scores to adversary. A trivial single token backdoor poisoning 1% of the evaluator training data triples the adversary's score with respect to their legitimate score. We systematically categorize levels of data access corresponding to three real-world settings, (1) web poisoning, (2) malicious annotator, and (3) weight poisoning. These regimes reflect a weak to strong escalation of data access that highly correlates with attack severity. Under the weakest assumptions - web poisoning (1), the adversary still induces a 20% score inflation. Likewise, in the (3) weight poisoning regime, the stronger assumptions enable the adversary to inflate their scores from 1.5/5 to 4.9/5. The backdoor threat generalizes across different evaluator architectures, trigger designs, evaluation tasks, and poisoning rates. By poisoning 10% of the evaluator training data, we control toxicity judges (Guardrails) to misclassify toxic prompts as non-toxic 89% of the time, and document reranker judges in RAG to rank the poisoned document first 97% of the time. LLM-as-a-Judge is uniquely positioned at the intersection of ethics and technology, where social implications of mislead model selection and evaluation constrain the available defensive tools. Amidst these challenges, model merging emerges as a principled tool to offset the backdoor, reducing ASR to near 0% whilst maintaining SOTA performance. Model merging's low computational cost and convenient integration into the current LLM Judge training pipeline position it as a promising avenue for backdoor mitigation in the LLM-as-a-Judge setting. 

---
# LoR2C : Low-Rank Residual Connection Adaptation for Parameter-Efficient Fine-Tuning 

**Authors**: Jiancheng Zhao, Xingda Yu, Yuxiang Zhang, Zhen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00572)  

**Abstract**: In recent years, pretrained large language models have demonstrated outstanding performance across various natural language processing tasks. However, full-parameter fine-tuning methods require adjusting all model parameters, leading to immense computational resource demands. Although parameter-efficient fine-tuning methods like LoRA have significantly reduced the number of parameters, they still face challenges such as gradient vanishing and the potential for further parameter reduction. To address these issues, this paper proposes a novel parameter-efficient fine-tuning method called LoR2C (Low-Rank Residual Connection Adaptation). LoR2C introduces residual connections with low-rank matrices within the model layers, which not only reduces the number of fine-tuning parameters but also effectively alleviates the gradient vanishing problem. Additionally, this paper presents three optimization variants of LoR2C: ShareLoR2C, MergeLoR2C, and InjectLoR2C. These variants further improve parameter efficiency and model performance through parameter sharing, module merging, and injection mechanisms, respectively. Experimental results on multiple natural language understanding and natural language generation tasks demonstrate that LoR2C and its optimized variants significantly reduce parameter overhead while maintaining or even improving performance, outperforming existing mainstream parameter-efficient fine-tuning this http URL code is publicly available at this https URL. 

---
# ToolDial: Multi-turn Dialogue Generation Method for Tool-Augmented Language Models 

**Authors**: Jeonghoon Shim, Gyuhyeon Seo, Cheongsu Lim, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2503.00564)  

**Abstract**: Tool-Augmented Language Models (TALMs) leverage external APIs to answer user queries across various domains. However, existing benchmark datasets for TALM research often feature simplistic dialogues that do not reflect real-world scenarios, such as the need for models to ask clarifying questions or proactively call additional APIs when essential information is missing. To address these limitations, we construct and release ToolDial, a dataset comprising 11,111 multi-turn dialogues, with an average of 8.95 turns per dialogue, based on APIs from RapidAPI. ToolDial has two key characteristics. First, the dialogues incorporate 16 user and system actions (e.g., "Request", "Clarify", "Fail inform") to capture the rich dynamics of real-world interactions. Second, we simulate dialogues where the system requests necessary information from the user based on API documentation and seeks additional APIs if the user fails to provide the required information. To facilitate this process, we introduce a method for generating an API graph that represents input and output compatibility between APIs. Using ToolDial, we evaluate a suite of language models on their ability to predict correct actions and extract input parameter values for API calls from the dialogue history. Modern language models achieve accuracy scores below 70%, indicating substantial room for improvement. We release our dataset and code at this https URL. 

---
# Tutorial Proposal: Speculative Decoding for Efficient LLM Inference 

**Authors**: Heming Xia, Cunxiao Du, Yongqi Li, Qian Liu, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00491)  

**Abstract**: This tutorial presents a comprehensive introduction to Speculative Decoding (SD), an advanced technique for LLM inference acceleration that has garnered significant research interest in recent years. SD is introduced as an innovative decoding paradigm to mitigate the high inference latency stemming from autoregressive decoding in LLMs. At each decoding step, SD efficiently drafts several future tokens and then verifies them in parallel. This approach, unlike traditional autoregressive decoding, facilitates the simultaneous decoding of multiple tokens per step, thereby achieving promising 2x-4x speedups in LLM inference while maintaining original distributions. This tutorial delves into the latest techniques in SD, including draft model architectures and verification strategies. Additionally, it explores the acceleration potential and future research directions in this promising field. We aim for this tutorial to elucidate the current research landscape and offer insights for researchers interested in Speculative Decoding, ultimately contributing to more efficient LLM inference. 

---
# Embracing Diversity: A Multi-Perspective Approach with Soft Labels 

**Authors**: Benedetta Muscato, Praveen Bushipaka, Gizem Gezici, Lucia Passaro, Fosca Giannotti, Tommaso Cucinotta  

**Link**: [PDF](https://arxiv.org/pdf/2503.00489)  

**Abstract**: Prior studies show that adopting the annotation diversity shaped by different backgrounds and life experiences and incorporating them into the model learning, i.e. multi-perspective approach, contribute to the development of more responsible models. Thus, in this paper we propose a new framework for designing and further evaluating perspective-aware models on stance detection task,in which multiple annotators assign stances based on a controversial topic. We also share a new dataset established through obtaining both human and LLM annotations. Results show that the multi-perspective approach yields better classification performance (higher F1-scores), outperforming the traditional approaches that use a single ground-truth, while displaying lower model confidence scores, probably due to the high level of subjectivity of the stance detection task. 

---
# Unstable Grounds for Beautiful Trees? Testing the Robustness of Concept Translations in the Compilation of Multilingual Wordlists 

**Authors**: David Snee, Luca Ciucci, Arne Rubehn, Kellen Parker van Dam, Johann-Mattis List  

**Link**: [PDF](https://arxiv.org/pdf/2503.00464)  

**Abstract**: Multilingual wordlists play a crucial role in comparative linguistics. While many studies have been carried out to test the power of computational methods for language subgrouping or divergence time estimation, few studies have put the data upon which these studies are based to a rigorous test. Here, we conduct a first experiment that tests the robustness of concept translation as an integral part of the compilation of multilingual wordlists. Investigating the variation in concept translations in independently compiled wordlists from 10 dataset pairs covering 9 different language families, we find that on average, only 83% of all translations yield the same word form, while identical forms in terms of phonetic transcriptions can only be found in 23% of all cases. Our findings can prove important when trying to assess the uncertainty of phylogenetic studies and the conclusions derived from them. 

---
# Rehearse With User: Personalized Opinion Summarization via Role-Playing based on Large Language Models 

**Authors**: Yanyue Zhang, Yulan He, Deyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.00449)  

**Abstract**: Personalized opinion summarization is crucial as it considers individual user interests while generating product summaries. Recent studies show that although large language models demonstrate powerful text summarization and evaluation capabilities without the need for training data, they face difficulties in personalized tasks involving long texts. To address this, \textbf{Rehearsal}, a personalized opinion summarization framework via LLMs-based role-playing is proposed. Having the model act as the user, the model can better understand the user's personalized needs. Additionally, a role-playing supervisor and practice process are introduced to improve the role-playing ability of the LLMs, leading to a better expression of user needs. Furthermore, through suggestions from virtual users, the summary generation is intervened, ensuring that the generated summary includes information of interest to the user, thus achieving personalized summary generation. Experiment results demonstrate that our method can effectively improve the level of personalization in large model-generated summaries. 

---
# Figurative Archive: an open dataset and web-based application for the study of metaphor 

**Authors**: Maddalena Bressler, Veronica Mangiaterra, Paolo Canal, Federico Frau, Fabrizio Luciani, Biagio Scalingi, Chiara Barattieri di San Pietro, Chiara Battaglini, Chiara Pompei, Fortunata Romeo, Luca Bischetti, Valentina Bambini  

**Link**: [PDF](https://arxiv.org/pdf/2503.00444)  

**Abstract**: Research on metaphor has steadily increased over the last decades, as this phenomenon opens a window into a range of processes in language and cognition, from pragmatic inference to abstraction and embodied simulation. At the same time, the demand for rigorously constructed and extensively normed experimental materials increased as well. Here, we present the Figurative Archive, an open database of 997 metaphors in Italian enriched with rating and corpus-based measures (from familiarity to lexical frequency), derived by collecting stimuli used across 11 studies. It includes both everyday and literary metaphors, varying in structure and semantic domains. Dataset validation comprised correlations between familiarity and other measures. The Figurative Archive has several aspects of novelty: it is increased in size compared to previous resources; it includes a novel measure of inclusiveness, to comply with current recommendations for non-discriminatory language use; it is displayed in a web-based interface, with features for a flexible and customized consultation. We provide guidelines for using the Archive in future metaphor studies, in the spirit of open science. 

---
# AILS-NTUA at SemEval-2025 Task 8: Language-to-Code prompting and Error Fixing for Tabular Question Answering 

**Authors**: Andreas Evangelatos, Giorgos Filandrianos, Maria Lymperaiou, Athanasios Voulodimos, Giorgos Stamou  

**Link**: [PDF](https://arxiv.org/pdf/2503.00435)  

**Abstract**: In this paper, we present our submission to SemEval-2025 Task 8: Question Answering over Tabular Data. This task, evaluated on the DataBench dataset, assesses Large Language Models' (LLMs) ability to answer natural language questions over structured data while addressing topic diversity and table size limitations in previous benchmarks. We propose a system that employs effective LLM prompting to translate natural language queries into executable code, enabling accurate responses, error correction, and interpretability. Our approach ranks first in both subtasks of the competition in the proprietary model category, significantly outperforming the organizer's baseline. 

---
# A Multi-Labeled Dataset for Indonesian Discourse: Examining Toxicity, Polarization, and Demographics Information 

**Authors**: Lucky Susanto, Musa Wijanarko, Prasetia Pratama, Zilu Tang, Fariz Akyas, Traci Hong, Ika Idris, Alham Aji, Derry Wijaya  

**Link**: [PDF](https://arxiv.org/pdf/2503.00417)  

**Abstract**: Polarization is defined as divisive opinions held by two or more groups on substantive issues. As the world's third-largest democracy, Indonesia faces growing concerns about the interplay between political polarization and online toxicity, which is often directed at vulnerable minority groups. Despite the importance of this issue, previous NLP research has not fully explored the relationship between toxicity and polarization. To bridge this gap, we present a novel multi-label Indonesian dataset that incorporates toxicity, polarization, and annotator demographic information. Benchmarking this dataset using BERT-base models and large language models (LLMs) shows that polarization information enhances toxicity classification, and vice versa. Furthermore, providing demographic information significantly improves the performance of polarization classification. 

---
# Smoothing Grounding and Reasoning for MLLM-Powered GUI Agents with Query-Oriented Pivot Tasks 

**Authors**: Zongru Wu, Pengzhou Cheng, Zheng Wu, Tianjie Ju, Zhuosheng Zhang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00401)  

**Abstract**: Perception-enhanced pre-training, particularly through grounding techniques, is widely adopted to enhance the performance of graphical user interface (GUI) agents. However, in resource-constrained scenarios, the format discrepancy between coordinate-oriented grounding and action-oriented reasoning limits the effectiveness of grounding for reasoning tasks. To address this challenge, we propose a query-oriented pivot approach called query inference, which serves as a bridge between GUI grounding and reasoning. By inferring potential user queries from a screenshot and its associated element coordinates, query inference improves the understanding of coordinates while aligning more closely with reasoning tasks. Experimental results show that query inference outperforms previous grounding techniques under the same training data scale. Notably, query inference achieves comparable or even better performance to large-scale grounding-enhanced OS-Atlas with less than 0.1% of training data. Furthermore, we explore the impact of reasoning formats and demonstrate that integrating additional semantic information into the input further boosts reasoning performance. The code is publicly available athttps://github.com/ZrW00/GUIPivot. 

---
# Approaching the Limits to EFL Writing Enhancement with AI-generated Text and Diverse Learners 

**Authors**: David James Woo, Hengky Susanto, Chi Ho Yeung, Kai Guo, Yilin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00367)  

**Abstract**: Generative artificial intelligence (AI) chatbots, such as ChatGPT, are reshaping how English as a foreign language (EFL) students write since students can compose texts by integrating their own words with AI-generated text. This study investigated how 59 Hong Kong secondary school students with varying levels of academic achievement interacted with AI-generated text to compose a feature article, exploring whether any interaction patterns benefited the overall quality of the article. Through content analysis, multiple linear regression and cluster analysis, we found the overall number of words -- whether AI- or human-generated -- is the main predictor of writing quality. However, the impact varies by students' competence to write independently, for instance, by using their own words accurately and coherently to compose a text, and to follow specific interaction patterns with AI-generated text. Therefore, although composing texts with human words and AI-generated text may become prevalent in EFL writing classrooms, without educators' careful attention to EFL writing pedagogy and AI literacy, high-achieving students stand to benefit more from using AI-generated text than low-achieving students. 

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
# Hierarchical Multi-Stage BERT Fusion Framework with Dual Attention for Enhanced Cyberbullying Detection in Social Media 

**Authors**: Jiani Wang, Xiaochuan Xu, Peiyang Yu, Zeqiu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00342)  

**Abstract**: Detecting and classifying cyberbullying in social media is hard because of the complex nature of online language and the changing nature of content. This study presents a multi-stage BERT fusion framework. It uses hierarchical embeddings, dual attention mechanisms, and extra features to improve detection of cyberbullying content. The framework combines BERT embeddings with features like sentiment and topic information. It uses self-attention and cross-attention to align features and has a hierarchical classification head for multi-category classification. A dynamic loss balancing strategy helps optimize learning and improves accuracy, precision, recall, and F1-score. These results show the model's strong performance and potential for broader use in analyzing social media content. 

---
# More of the Same: Persistent Representational Harms Under Increased Representation 

**Authors**: Jennifer Mickel, Maria De-Arteaga, Leqi Liu, Kevin Tian  

**Link**: [PDF](https://arxiv.org/pdf/2503.00333)  

**Abstract**: To recognize and mitigate the harms of generative AI systems, it is crucial to consider who is represented in the outputs of generative AI systems and how people are represented. A critical gap emerges when naively improving who is represented, as this does not imply bias mitigation efforts have been applied to address how people are represented. We critically examined this by investigating gender representation in occupation across state-of-the-art large language models. We first show evidence suggesting that over time there have been interventions to models altering the resulting gender distribution, and we find that women are more represented than men when models are prompted to generate biographies or personas. We then demonstrate that representational biases persist in how different genders are represented by examining statistically significant word differences across genders. This results in a proliferation of representational harms, stereotypes, and neoliberalism ideals that, despite existing interventions to increase female representation, reinforce existing systems of oppression. 

---
# How Deep is Love in LLMs' Hearts? Exploring Semantic Size in Human-like Cognition 

**Authors**: Yao Yao, Yifei Yang, Xinbei Ma, Dongjie Yang, Zhuosheng Zhang, Zuchao Li, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.00330)  

**Abstract**: How human cognitive abilities are formed has long captivated researchers. However, a significant challenge lies in developing meaningful methods to measure these complex processes. With the advent of large language models (LLMs), which now rival human capabilities in various domains, we are presented with a unique testbed to investigate human cognition through a new lens. Among the many facets of cognition, one particularly crucial aspect is the concept of semantic size, the perceived magnitude of both abstract and concrete words or concepts. This study seeks to investigate whether LLMs exhibit similar tendencies in understanding semantic size, thereby providing insights into the underlying mechanisms of human cognition. We begin by exploring metaphorical reasoning, comparing how LLMs and humans associate abstract words with concrete objects of varying sizes. Next, we examine LLMs' internal representations to evaluate their alignment with human cognitive processes. Our findings reveal that multi-modal training is crucial for LLMs to achieve more human-like understanding, suggesting that real-world, multi-modal experiences are similarly vital for human cognitive development. Lastly, we examine whether LLMs are influenced by attention-grabbing headlines with larger semantic sizes in a real-world web shopping scenario. The results show that multi-modal LLMs are more emotionally engaged in decision-making, but this also introduces potential biases, such as the risk of manipulation through clickbait headlines. Ultimately, this study offers a novel perspective on how LLMs interpret and internalize language, from the smallest concrete objects to the most profound abstract concepts like love. The insights gained not only improve our understanding of LLMs but also provide new avenues for exploring the cognitive abilities that define human intelligence. 

---
# Unlocking Efficient, Scalable, and Continual Knowledge Editing with Basis-Level Representation Fine-Tuning 

**Authors**: Tianci Liu, Ruirui Li, Yunzhe Qi, Hui Liu, Xianfeng Tang, Tianqi Zheng, Qingyu Yin, Monica Xiao Cheng, Jun Huan, Haoyu Wang, Jing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.00306)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance on various natural language tasks. However, they are trained on static corpora and their knowledge can become outdated quickly in the fast-changing world. This motivates the development of knowledge editing methods designed to update certain knowledge in LLMs without changing unrelated others. To make selective edits, previous efforts often sought to update a small amount of parameters in some specific layer(s) of a LLM. Nonetheless, in challenging scenarios, they still fall short in making successful edits while preserving knowledge irrelevant to the updates simultaneously, resulting in a notable editing-locality trade-off. In this work, we question if the trade-offs are caused by the fact that parameter-based updates have a global effect, i.e., edited parameters affect all inputs indiscriminately. In light of this, we explore the feasibility of representation fine-tuning, which applied some linear update to a few representations in a learned subspace, for knowledge editing. While being effective to enhance an LLM's general ability as demonstrated in the previous work, we theoretically show that this linear update imposes a tension in editing-locality trade-off. Subsequently, BaFT is proposed to break the linearity. BaFT computes a weight for each basis that spans a dimension of the subspace based on the input representation. This input-dependent weighting mechanism allows BaFT to manage different types of knowledge in an adaptive way, thereby achieving a better editing-locality trade-off. Experiments on three LLMs with five editing benchmarks in diverse scenarios show the superiority of our method. 

---
# Robust Multi-Objective Preference Alignment with Online DPO 

**Authors**: Raghav Gupta, Ryan Sullivan, Yunxuan Li, Samrat Phatale, Abhinav Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2503.00295)  

**Abstract**: Multi-objective preference alignment of large language models (LLMs) is critical for developing AI systems that are more configurable, personalizable, helpful, and safe. However, optimizing model outputs to satisfy diverse objectives with variable weights at inference time for truly personalized models presents a significant challenge. Existing approaches are either computationally expensive to train or do not sufficiently steer model behaviors. This paper introduces the Multi-Objective Online DPO (MO-ODPO) algorithm, designed to robustly and efficiently align model behaviors with multiple, potentially conflicting human preferences. Our approach incorporates a prompt conditioning mechanism, allowing us to train a single preference-conditional policy, that can adapt to new preference combinations at inference. Experiments on two popular benchmarks show that MO-ODPO Pareto-dominates existing baselines while providing excellent inference-time steerability between diverse objectives. 

---
# Decoupling Content and Expression: Two-Dimensional Detection of AI-Generated Text 

**Authors**: Guangsheng Bao, Lihua Rong, Yanbin Zhao, Qiji Zhou, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00258)  

**Abstract**: The wide usage of LLMs raises critical requirements on detecting AI participation in texts. Existing studies investigate these detections in scattered contexts, leaving a systematic and unified approach unexplored. In this paper, we present HART, a hierarchical framework of AI risk levels, each corresponding to a detection task. To address these tasks, we propose a novel 2D Detection Method, decoupling a text into content and language expression. Our findings show that content is resistant to surface-level changes, which can serve as a key feature for detection. Experiments demonstrate that 2D method significantly outperforms existing detectors, achieving an AUROC improvement from 0.705 to 0.849 for level-2 detection and from 0.807 to 0.886 for RAID. We release our data and code at this https URL. 

---
# Jawaher: A Multidialectal Dataset of Arabic Proverbs for LLM Benchmarking 

**Authors**: Samar M. Magdy, Sang Yun Kwon, Fakhraddin Alwajih, Safaa Abdelfadil, Shady Shehata, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2503.00231)  

**Abstract**: Recent advancements in instruction fine-tuning, alignment methods such as reinforcement learning from human feedback (RLHF), and optimization techniques like direct preference optimization (DPO) have significantly enhanced the adaptability of large language models (LLMs) to user preferences. However, despite these innovations, many LLMs continue to exhibit biases toward Western, Anglo-centric, or American cultures, with performance on English data consistently surpassing that of other languages. This reveals a persistent cultural gap in LLMs, which complicates their ability to accurately process culturally rich and diverse figurative language such as proverbs. To address this, we introduce Jawaher, a benchmark designed to assess LLMs' capacity to comprehend and interpret Arabic proverbs. Jawaher includes proverbs from various Arabic dialects, along with idiomatic translations and explanations. Through extensive evaluations of both open- and closed-source models, we find that while LLMs can generate idiomatically accurate translations, they struggle with producing culturally nuanced and contextually relevant explanations. These findings highlight the need for ongoing model refinement and dataset expansion to bridge the cultural gap in figurative language processing. 

---
# À la recherche du sens perdu: your favourite LLM might have more to say than you can understand 

**Authors**: K. O. T. Erziev  

**Link**: [PDF](https://arxiv.org/pdf/2503.00224)  

**Abstract**: We report a peculiar observation that LLMs can assign hidden meanings to sequences that seem visually incomprehensible to humans: for example, a nonsensical phrase consisting of Byzantine musical symbols is recognized by gpt-4o as "say abracadabra". Moreover, some models can communicate using these sequences.
Some of these meanings are hypothesized to partly originate in the massive spurious correlations due to BPE tokenization. We systematically evaluate the presence of such abilities in a wide range of models: Claude-3.5 Haiku, Claude-3.5 Sonnet (New and Old), Claude-3.7 Sonnet, gpt-4o mini, gpt-4o, o1-mini, Llama-3.3 70B, DeepSeek-R1-Distill-Lllama 70B, Qwen2.5 1.5B, Qwen2.5 32B, Phi-3.5 mini, GigaChat-Max, Vikhr-Llama-3.2 1B.
We argue that this observation might have far-reaching consequences for both safety and security of the modern and future LLMs and systems that employ them. As an illustration, we show that applying this method in combination with simple templates is sufficient to jailbreak previous generation models, with ASR = 0.4 on gpt-4o mini.
Our code and data artifacts are available at this https URL 

---
# Autoencoder-Based Framework to Capture Vocabulary Quality in NLP 

**Authors**: Vu Minh Hoang Dang, Rakesh M. Verma  

**Link**: [PDF](https://arxiv.org/pdf/2503.00209)  

**Abstract**: Linguistic richness is essential for advancing natural language processing (NLP), as dataset characteristics often directly influence model performance. However, traditional metrics such as Type-Token Ratio (TTR), Vocabulary Diversity (VOCD), and Measure of Lexical Text Diversity (MTLD) do not adequately capture contextual relationships, semantic richness, and structural complexity. In this paper, we introduce an autoencoder-based framework that uses neural network capacity as a proxy for vocabulary richness, diversity, and complexity, enabling a dynamic assessment of the interplay between vocabulary size, sentence structure, and contextual depth. We validate our approach on two distinct datasets: the DIFrauD dataset, which spans multiple domains of deceptive and fraudulent text, and the Project Gutenberg dataset, representing diverse languages, genres, and historical periods. Experimental results highlight the robustness and adaptability of our method, offering practical guidance for dataset curation and NLP model design. By enhancing traditional vocabulary evaluation, our work fosters the development of more context-aware, linguistically adaptive NLP systems. 

---
# Llamarine: Open-source Maritime Industry-specific Large Language Model 

**Authors**: William Nguyen, An Phan, Konobu Kimura, Hitoshi Maeno, Mika Tanaka, Quynh Le, William Poucher, Christopher Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2503.00203)  

**Abstract**: Large Language Models (LLMs) have demonstrated substantial potential in addressing complex reasoning tasks, yet their general-purpose nature often limits their effectiveness in specialized domains such as maritime navigation. To bridge this gap, we introduce Llamarine, the first open-source LLM designed specifically for maritime navigation. Llamarine 1.0 is developed through continued pretraining and fine-tuning on a high-quality corpus comprising maritime textbooks, research publications, and web text from Wikipedia. This domain-specific training enables the model to acquire expert-level knowledge in navigational principles, collision avoidance, route optimization, and regulatory compliance. Our key contributions include (a) the curation of a comprehensive maritime dataset from authoritative sources, ensuring depth and reliability in the model's knowledge base; (b) the development of a foundational model capable of reasoning about complex navigational challenges with greater accuracy than general-purpose LLMs; and (c) the establishment of a benchmark to evaluate performance in maritime-specific decision-making tasks. Experimental results demonstrate that Llamarine outperforms both general-purpose and commercial LLMs in critical navigation-related tasks, such as trajectory planning, risk assessment, and compliance with maritime regulations. By providing an open-source foundation model trained exclusively on high-quality maritime literature, Llamarine paves the way for AI-driven advancements in maritime safety, efficiency, and operational decision-making. 

---
# A Survey of Uncertainty Estimation Methods on Large Language Models 

**Authors**: Zhiqiu Xia, Jinxuan Xu, Yuqian Zhang, Hang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00172)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various tasks. However, these models could offer biased, hallucinated, or non-factual responses camouflaged by their fluency and realistic appearance. Uncertainty estimation is the key method to address this challenge. While research efforts in uncertainty estimation are ramping up, there is a lack of comprehensive and dedicated surveys on LLM uncertainty estimation. This survey presents four major avenues of LLM uncertainty estimation. Furthermore, we perform extensive experimental evaluations across multiple methods and datasets. At last, we provide critical and promising future directions for LLM uncertainty estimation. 

---
# Palm: A Culturally Inclusive and Linguistically Diverse Dataset for Arabic LLMs 

**Authors**: Fakhraddin Alwajih, Abdellah El Mekki, Samar Mohamed Magdy, Abdelrahim A. Elmadany, Omer Nacar, El Moatez Billah Nagoudi, Reem Abdel-Salam, Hanin Atwany, Youssef Nafea, Abdulfattah Mohammed Yahya, Rahaf Alhamouri, Hamzah A. Alsayadi, Hiba Zayed, Sara Shatnawi, Serry Sibaee, Yasir Ech-Chammakhy, Walid Al-Dhabyani, Marwa Mohamed Ali, Imen Jarraya, Ahmed Oumar El-Shangiti, Aisha Alraeesi, Mohammed Anwar Al-Ghrawi, Abdulrahman S. Al-Batati, Elgizouli Mohamed, Noha Taha Elgindi, Muhammed Saeed, Houdaifa Atou, Issam Ait Yahia, Abdelhak Bouayad, Mohammed Machrouh, Amal Makouar, Dania Alkawi, Mukhtar Mohamed, Safaa Taher Abdelfadil, Amine Ziad Ounnoughene, Rouabhia Anfel, Rwaa Assi, Ahmed Sorkatti, Mohamedou Cheikh Tourad, Anis Koubaa, Ismail Berrada, Mustafa Jarrar, Shady Shehata, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2503.00151)  

**Abstract**: As large language models (LLMs) become increasingly integrated into daily life, ensuring their cultural sensitivity and inclusivity is paramount. We introduce our dataset, a year-long community-driven project covering all 22 Arab countries. The dataset includes instructions (input, response pairs) in both Modern Standard Arabic (MSA) and dialectal Arabic (DA), spanning 20 diverse topics. Built by a team of 44 researchers across the Arab world, all of whom are authors of this paper, our dataset offers a broad, inclusive perspective. We use our dataset to evaluate the cultural and dialectal capabilities of several frontier LLMs, revealing notable limitations. For instance, while closed-source LLMs generally exhibit strong performance, they are not without flaws, and smaller open-source models face greater challenges. Moreover, certain countries (e.g., Egypt, the UAE) appear better represented than others (e.g., Iraq, Mauritania, Yemen). Our annotation guidelines, code, and data for reproducibility are publicly available. 

---
# SCORE: Systematic COnsistency and Robustness Evaluation for Large Language Models 

**Authors**: Grigor Nalbandyan, Rima Shahbazyan, Evelina Bakhturina  

**Link**: [PDF](https://arxiv.org/pdf/2503.00137)  

**Abstract**: Typical evaluations of Large Language Models (LLMs) report a single metric per dataset, often representing the model's best-case performance under carefully selected settings. Unfortunately, this approach overlooks model robustness and reliability in real-world applications. For instance, simple paraphrasing of prompts on the MMLU-Pro dataset causes accuracy fluctuations of up to 10\%, while reordering answer choices in the AGIEval dataset results in accuracy differences of up to 6.1\%. While some studies discuss issues with LLM robustness, there is no unified or centralized framework for evaluating the robustness of language models. To address this gap and consolidate existing research on model robustness, we present SCORE ($\mathbf{S}$ystematic $\mathbf{CO}$nsistency and $\mathbf{R}$obustness $\mathbf{E}$valuation), a comprehensive framework for non-adversarial evaluation of LLMs. The SCORE framework evaluates models by repeatedly testing them on the same benchmarks in various setups to give a realistic estimate of their accuracy and consistency. We release the code publicly and start an LLM robustness leaderboard to facilitate further development and research. 

---
# Personalized Causal Graph Reasoning for LLMs: A Case Study on Dietary Recommendations 

**Authors**: Zhongqi Yang, Amir Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2503.00134)  

**Abstract**: Large Language Models (LLMs) effectively leverage common-sense knowledge for general reasoning, yet they struggle with personalized reasoning when tasked with interpreting multifactor personal data. This limitation restricts their applicability in domains that require context-aware decision-making tailored to individuals. This paper introduces Personalized Causal Graph Reasoning as an agentic framework that enhances LLM reasoning by incorporating personal causal graphs derived from data of individuals. These graphs provide a foundation that guides the LLM's reasoning process. We evaluate it on a case study on nutrient-oriented dietary recommendations, which requires personal reasoning due to the implicit unique dietary effects. We propose a counterfactual evaluation to estimate the efficiency of LLM-recommended foods for glucose management. Results demonstrate that the proposed method efficiently provides personalized dietary recommendations to reduce average glucose iAUC across three time windows, which outperforms the previous approach. LLM-as-a-judge evaluation results indicate that our proposed method enhances personalization in the reasoning process. 

---
# AnnoCaseLaw: A Richly-Annotated Dataset For Benchmarking Explainable Legal Judgment Prediction 

**Authors**: Magnus Sesodia, Alina Petrova, John Armour, Thomas Lukasiewicz, Oana-Maria Camburu, Puneet K. Dokania, Philip Torr, Christian Schroeder de Witt  

**Link**: [PDF](https://arxiv.org/pdf/2503.00128)  

**Abstract**: Legal systems worldwide continue to struggle with overwhelming caseloads, limited judicial resources, and growing complexities in legal proceedings. Artificial intelligence (AI) offers a promising solution, with Legal Judgment Prediction (LJP) -- the practice of predicting a court's decision from the case facts -- emerging as a key research area. However, existing datasets often formulate the task of LJP unrealistically, not reflecting its true difficulty. They also lack high-quality annotation essential for legal reasoning and explainability. To address these shortcomings, we introduce AnnoCaseLaw, a first-of-its-kind dataset of 471 meticulously annotated U.S. Appeals Court negligence cases. Each case is enriched with comprehensive, expert-labeled annotations that highlight key components of judicial decision making, along with relevant legal concepts. Our dataset lays the groundwork for more human-aligned, explainable LJP models. We define three legally relevant tasks: (1) judgment prediction; (2) concept identification; and (3) automated case annotation, and establish a performance baseline using industry-leading large language models (LLMs). Our results demonstrate that LJP remains a formidable task, with application of legal precedent proving particularly difficult. Code and data are available at this https URL. 

---
# Evaluation of LLMs-based Hidden States as Author Representations for Psychological Human-Centered NLP Tasks 

**Authors**: Nikita Soni, Pranav Chitale, Khushboo Singh, Niranjan Balasubramanian, H. Andrew Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2503.00124)  

**Abstract**: Like most of NLP, models for human-centered NLP tasks -- tasks attempting to assess author-level information -- predominantly use representations derived from hidden states of Transformer-based LLMs. However, what component of the LM is used for the representation varies widely. Moreover, there is a need for Human Language Models (HuLMs) that implicitly model the author and provide a user-level hidden state. Here, we systematically evaluate different ways of representing documents and users using different LM and HuLM architectures to predict task outcomes as both dynamically changing states and averaged trait-like user-level attributes of valence, arousal, empathy, and distress. We find that representing documents as an average of the token hidden states performs the best generally. Further, while a user-level hidden state itself is rarely the best representation, we find its inclusion in the model strengthens token or document embeddings used to derive document- and user-level representations resulting in best performances. 

---
# from Benign import Toxic: Jailbreaking the Language Model via Adversarial Metaphors 

**Authors**: Yu Yan, Sheng Sun, Zenghao Duan, Teli Liu, Min Liu, Zhiyi Yin, Qi Li, Jiangyu Lei  

**Link**: [PDF](https://arxiv.org/pdf/2503.00038)  

**Abstract**: Current studies have exposed the risk of Large Language Models (LLMs) generating harmful content by jailbreak attacks. However, they overlook that the direct generation of harmful content from scratch is more difficult than inducing LLM to calibrate benign content into harmful forms. In our study, we introduce a novel attack framework that exploits AdVersArial meTAphoR (AVATAR) to induce the LLM to calibrate malicious metaphors for jailbreaking. Specifically, to answer harmful queries, AVATAR adaptively identifies a set of benign but logically related metaphors as the initial seed. Then, driven by these metaphors, the target LLM is induced to reason and calibrate about the metaphorical content, thus jailbroken by either directly outputting harmful responses or calibrating residuals between metaphorical and professional harmful content. Experimental results demonstrate that AVATAR can effectively and transferable jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs. 

---
# Zero-Shot Defense Against Toxic Images via Inherent Multimodal Alignment in LVLMs 

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.00037)  

**Abstract**: Large Vision-Language Models (LVLMs) have made significant strides in multimodal comprehension, thanks to extensive pre-training and fine-tuning on large-scale visual datasets. However, despite their robust textual safety mechanisms, they remain vulnerable to harmful visual inputs. Existing safeguards-typically relying on pre-filtering or fine-tuning-incur high costs and diminish overall utility. To address this critical vulnerability, we introduce SafeCLIP, a lightweight method that leverages LVLMs inherent multimodal alignment for zero-shot toxic image detection. By projecting CLIPs discarded CLS token into its text space and matching it with toxic descriptors, SafeCLIP detects harmful content without any architectural changes-adding minimal latency and enabling dynamic safety corrections during inference and this http URL show that SafeCLIP achieves a 66.9% defense success rate with only 3.2% false positive rate and 7.2% overhead. In contrast, state-of-the-art methods achieve 52.9% success but have a 10.7% false positive rate and 210% overhead. Our work demonstrates that leveraging inherent multimodal alignment can yield efficient, low-cost LVLM safety. Code is available at this http URL. 

---
# Constraining Sequential Model Editing with Editing Anchor Compression 

**Authors**: Hao-Xiang Xu, Jun-Yu Ma, Zhen-Hua Ling, Ningyu Zhang, Jia-Chen Gu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00035)  

**Abstract**: Large language models (LLMs) struggle with hallucinations due to false or outdated knowledge. Given the high resource demands of retraining these models, there is an increasing focus on developing model editing. However, the general abilities of LLMs across downstream tasks are prone to significant degradation during sequential editing. This paper statistically observes that the parameter matrix after editing exhibits a significant deviation compared to its previous state as the number of edits increases. This serious deviation affects the original knowledge associations within LLMs and leads to the degradation of their general abilities. To this end, a framework termed Editing Anchor Compression (EAC) is proposed to constrain the deviation of the parameter matrix during sequential editing. It compresses the editing information by selecting editing anchors that are important in encoding new relations without deviating too much from the original matrix, thereby preserving the general abilities. Experiments of applying EAC to two popular editing methods on three LLMs across four tasks are conducted. Evaluation results show that EAC effectively minimizes unreasonable deviations caused by model editing, preserving over 70% of the general abilities while better retaining the editing knowledge compared to the original counterpart methods. 

---
# Detecting LLM-Generated Korean Text through Linguistic Feature Analysis 

**Authors**: Shinwoo Park, Shubin Kim, Do-Kyung Kim, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.00032)  

**Abstract**: The rapid advancement of large language models (LLMs) increases the difficulty of distinguishing between human-written and LLM-generated text. Detecting LLM-generated text is crucial for upholding academic integrity, preventing plagiarism, protecting copyrights, and ensuring ethical research practices. Most prior studies on detecting LLM-generated text focus primarily on English text. However, languages with distinct morphological and syntactic characteristics require specialized detection approaches. Their unique structures and usage patterns can hinder the direct application of methods primarily designed for English. Among such languages, we focus on Korean, which has relatively flexible spacing rules, a rich morphological system, and less frequent comma usage compared to English. We introduce KatFish, the first benchmark dataset for detecting LLM-generated Korean text. The dataset consists of text written by humans and generated by four LLMs across three genres.
By examining spacing patterns, part-of-speech diversity, and comma usage, we illuminate the linguistic differences between human-written and LLM-generated Korean text. Building on these observations, we propose KatFishNet, a detection method specifically designed for the Korean language. KatFishNet achieves an average of 19.78% higher AUROC compared to the best-performing existing detection method. 

---
# Evaluating Large Language Models on the Spanish Medical Intern Resident (MIR) Examination 2024/2025:A Comparative Analysis of Clinical Reasoning and Knowledge Application 

**Authors**: Carlos Luengo Vera, Ignacio Ferro Picon, M. Teresa del Val Nunez, Jose Andres Gomez Gandia, Antonio de Lucas Ancillo, Victor Ramos Arroyo, Carlos Milan Figueredo  

**Link**: [PDF](https://arxiv.org/pdf/2503.00025)  

**Abstract**: This study presents a comparative evaluation of 22 large language models LLMs on the Spanish Medical Intern Resident MIR examinations for 2024 and 2025 with a focus on clinical reasoning domain specific expertise and multimodal processing capabilities The MIR exam consisting of 210 multiple choice questions some requiring image interpretation serves as a stringent benchmark for assessing both factual recall and complex clinical problem solving skills Our investigation encompasses general purpose models such as GPT4 Claude LLaMA and Gemini as well as specialized fine tuned systems like Miri Pro which leverages proprietary Spanish healthcare data to excel in medical contexts
Recent market entries Deepseek and Grok have further enriched the evaluation landscape particularly for tasks that demand advanced visual and semantic analysis The findings indicate that while general purpose LLMs perform robustly overall fine tuned models consistently achieve superior accuracy especially in addressing nuanced domain specific challenges A modest performance decline observed between the two exam cycles appears attributable to the implementation of modified questions designed to mitigate reliance on memorization
The results underscore the transformative potential of domain specific fine tuning and multimodal integration in advancing medical AI applications They also highlight critical implications for the future integration of LLMs into medical education training and clinical decision making emphasizing the importance of balancing automated reasoning with ethical and context aware judgment 

---
# Do Emotions Really Affect Argument Convincingness? A Dynamic Approach with LLM-based Manipulation Checks 

**Authors**: Yanran Chen, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2503.00024)  

**Abstract**: Emotions have been shown to play a role in argument convincingness, yet this aspect is underexplored in the natural language processing (NLP) community. Unlike prior studies that use static analyses, focus on a single text domain or language, or treat emotion as just one of many factors, we introduce a dynamic framework inspired by manipulation checks commonly used in psychology and social science; leveraging LLM-based manipulation checks, this framework examines the extent to which perceived emotional intensity influences perceived convincingness. Through human evaluation of arguments across different languages, text domains, and topics, we find that in over half of cases, judgments of convincingness remain unchanged despite variations in perceived emotional intensity; when emotions do have an impact, they more often enhance rather than weaken convincingness. We further analyze how 11 LLMs behave in the same scenario, finding that while LLMs generally mirror human patterns, they struggle to capture nuanced emotional effects in individual judgments. 

---
# KVCrush: Key value cache size-reduction using similarity in head-behaviour 

**Authors**: Gopi Krishna Jha, Sameh Gobriel, Liubov Talamanova, Alexander Kozlov, Nilesh Jain  

**Link**: [PDF](https://arxiv.org/pdf/2503.00022)  

**Abstract**: Key-value (KV) caching has emerged as a crucial optimization technique for accelerating inference in large language models (LLMs). By allowing the attention operation to scale linearly rather than quadratically with the total sequence length, KV caching significantly enhances generation throughput. However, due to large context lengths in the modern LLMs, the memory footprint of the KV is a huge bottleneck for model deployment directly impacting the model's batch size, hindering its ability to deliver high-throughput. Existing research addresses this challenge using several techniques, such as discarding low-attention tokens, quantization, and matrix approximation which typically lead to a negative impact on the model accuracy.
In this paper, We propose KVCrush technology which can be combined with many KV compression technologies to improve the model accuracy at a much smaller memory. KVCrush provides an alternate representation scheme for key-value states, along with a low-overhead token pruning algorithm that accounts for the token distribution in the KV cache, which in turn allows for a a smaller footprint while maintaining the accuracy of the model. Based on our results, KVCrush reduces LongBench KV Cache size by 4x with less than 1% accuracy drop and achieves state-of-the-art average accuracy with minimal overhead, incurring less than 0.5% total inference latency. KVCrush not only outperforms the accuracy of state-of-the-art importance-based token retention schemes but is also compatible with typical practical LLM deployments using KV cache paging schemes such as vLLM and mixed precision quantization. 

---
# A Systematic Review of Open Datasets Used in Text-to-Image (T2I) Gen AI Model Safety 

**Authors**: Rakeen Rouf, Trupti Bavalatti, Osama Ahmed, Dhaval Potdar, Faraz Jawed  

**Link**: [PDF](https://arxiv.org/pdf/2503.00020)  

**Abstract**: Novel research aimed at text-to-image (T2I) generative AI safety often relies on publicly available datasets for training and evaluation, making the quality and composition of these datasets crucial. This paper presents a comprehensive review of the key datasets used in the T2I research, detailing their collection methods, compositions, semantic and syntactic diversity of prompts and the quality, coverage, and distribution of harm types in the datasets. By highlighting the strengths and limitations of the datasets, this study enables researchers to find the most relevant datasets for a use case, critically assess the downstream impacts of their work given the dataset distribution, particularly regarding model safety and ethical considerations, and also identify the gaps in dataset coverage and quality that future research may address. 

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
# RSQ: Learning from Important Tokens Leads to Better Quantized LLMs 

**Authors**: Yi-Lin Sung, Prateek Yadav, Jialu Li, Jaehong Yoon, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2503.01820)  

**Abstract**: Layer-wise quantization is a key technique for efficiently compressing large models without expensive retraining. Previous methods typically quantize the weights of each layer by "uniformly" optimizing the layer reconstruction loss across all output tokens. However, in this paper, we demonstrate that better-quantized models can be obtained by prioritizing learning from important tokens (e.g. which have large attention scores). Building on this finding, we propose RSQ (Rotate, Scale, then Quantize), which (1) applies rotations (orthogonal transformation) to the model to mitigate outliers (those with exceptionally large magnitude), (2) scales the token feature based on its importance, and (3) quantizes the model using the GPTQ framework with the second-order statistics computed by scaled tokens. To compute token importance, we explore both heuristic and dynamic strategies. Based on a thorough analysis of all approaches, we adopt attention concentration, which uses attention scores of each token as its importance, as the best approach. We demonstrate that RSQ consistently outperforms baseline methods across multiple downstream tasks and three model families: LLaMA3, Mistral, and Qwen2.5. Additionally, models quantized with RSQ achieve superior performance on long-context tasks, further highlighting its effectiveness. Lastly, RSQ demonstrates generalizability across various setups, including different model sizes, calibration datasets, bit precisions, and quantization methods. 

---
# Do GFlowNets Transfer? Case Study on the Game of 24/42 

**Authors**: Adesh Gupta, Abhinav Kumar, Mansi Gupta, Paras Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2503.01819)  

**Abstract**: Generating diverse solutions is key to human-like reasoning, yet autoregressive language models focus on single accurate responses, limiting creativity. GFlowNets optimize solution generation as a flow network, promising greater diversity. Our case study shows their limited zero-shot transferability by fine-tuning small and medium-sized large language models on the Game of 24 and testing them on the Game of 42 datasets. Results revealed that GFlowNets struggle to maintain solution diversity and accuracy, highlighting key limitations in their cross-task generalization and the need for future research in improved transfer learning capabilities. 

---
# LLMInit: A Free Lunch from Large Language Models for Selective Initialization of Recommendation 

**Authors**: Weizhi Zhang, Liangwei Yang, Wooseong Yang, Henry Peng Zou, Yuqing Liu, Ke Xu, Sourav Medya, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01814)  

**Abstract**: Collaborative filtering models, particularly graph-based approaches, have demonstrated strong performance in capturing user-item interactions for recommendation systems. However, they continue to struggle in cold-start and data-sparse scenarios. The emergence of large language models (LLMs) like GPT and LLaMA presents new possibilities for enhancing recommendation performance, especially in cold-start settings. Despite their promise, LLMs pose challenges related to scalability and efficiency due to their high computational demands and limited ability to model complex user-item relationships effectively. In this work, we introduce a novel perspective on leveraging LLMs for CF model initialization. Through experiments, we uncover an embedding collapse issue when scaling CF models to larger embedding dimensions. To effectively harness large-scale LLM embeddings, we propose innovative selective initialization strategies utilizing random, uniform, and variance-based index sampling. Our comprehensive evaluation on multiple real-world datasets demonstrates significant performance gains across various CF models while maintaining a lower computational cost compared to existing LLM-based recommendation approaches. 

---
# Depth-Width tradeoffs in Algorithmic Reasoning of Graph Tasks with Transformers 

**Authors**: Gilad Yehudai, Clayton Sanford, Maya Bechler-Speicher, Orr Fischer, Ran Gilad-Bachrach, Amir Globerson  

**Link**: [PDF](https://arxiv.org/pdf/2503.01805)  

**Abstract**: Transformers have revolutionized the field of machine learning. In particular, they can be used to solve complex algorithmic problems, including graph-based tasks. In such algorithmic tasks a key question is what is the minimal size of a transformer that can implement a task. Recent work has begun to explore this problem for graph-based tasks, showing that for sub-linear embedding dimension (i.e., model width) logarithmic depth suffices. However, an open question, which we address here, is what happens if width is allowed to grow linearly. Here we analyze this setting, and provide the surprising result that with linear width, constant depth suffices for solving a host of graph-based problems. This suggests that a moderate increase in width can allow much shallower models, which are advantageous in terms of inference time. For other problems, we show that quadratic width is required. Our results demonstrate the complex and intriguing landscape of transformer implementations of graph-based algorithms. We support our theoretical results with empirical evaluations. 

---
# SAKE: Steering Activations for Knowledge Editing 

**Authors**: Marco Scialanga, Thibault Laugel, Vincent Grari, Marcin Detyniecki  

**Link**: [PDF](https://arxiv.org/pdf/2503.01751)  

**Abstract**: As Large Langue Models have been shown to memorize real-world facts, the need to update this knowledge in a controlled and efficient manner arises. Designed with these constraints in mind, Knowledge Editing (KE) approaches propose to alter specific facts in pretrained models. However, they have been shown to suffer from several limitations, including their lack of contextual robustness and their failure to generalize to logical implications related to the fact. To overcome these issues, we propose SAKE, a steering activation method that models a fact to be edited as a distribution rather than a single prompt. Leveraging Optimal Transport, SAKE alters the LLM behavior over a whole fact-related distribution, defined as paraphrases and logical implications. Several numerical experiments demonstrate the effectiveness of this method: SAKE is thus able to perform more robust edits than its existing counterparts. 

---
# MAPS: Motivation-Aware Personalized Search via LLM-Driven Consultation Alignment 

**Authors**: Weicong Qin, Yi Xu, Weijie Yu, Chenglei Shen, Ming He, Jianping Fan, Xiao Zhang, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01711)  

**Abstract**: Personalized product search aims to retrieve and rank items that match users' preferences and search intent. Despite their effectiveness, existing approaches typically assume that users' query fully captures their real motivation. However, our analysis of a real-world e-commerce platform reveals that users often engage in relevant consultations before searching, indicating they refine intents through consultations based on motivation and need. The implied motivation in consultations is a key enhancing factor for personalized search. This unexplored area comes with new challenges including aligning contextual motivations with concise queries, bridging the category-text gap, and filtering noise within sequence history. To address these, we propose a Motivation-Aware Personalized Search (MAPS) method. It embeds queries and consultations into a unified semantic space via LLMs, utilizes a Mixture of Attention Experts (MoAE) to prioritize critical semantics, and introduces dual alignment: (1) contrastive learning aligns consultations, reviews, and product features; (2) bidirectional attention integrates motivation-aware embeddings with user preferences. Extensive experiments on real and synthetic data show MAPS outperforms existing methods in both retrieval and ranking tasks. 

---
# Code-as-Symbolic-Planner: Foundation Model-Based Robot Planning via Symbolic Code Generation 

**Authors**: Yongchao Chen, Yilun Hao, Yang Zhang, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.01700)  

**Abstract**: Recent works have shown great potentials of Large Language Models (LLMs) in robot task and motion planning (TAMP). Current LLM approaches generate text- or code-based reasoning chains with sub-goals and action plans. However, they do not fully leverage LLMs' symbolic computing and code generation capabilities. Many robot TAMP tasks involve complex optimization under multiple constraints, where pure textual reasoning is insufficient. While augmenting LLMs with predefined solvers and planners improves performance, it lacks generalization across tasks. Given LLMs' growing coding proficiency, we enhance their TAMP capabilities by steering them to generate code as symbolic planners for optimization and constraint verification. Unlike prior work that uses code to interface with robot action modules, we steer LLMs to generate code as solvers, planners, and checkers for TAMP tasks requiring symbolic computing, while still leveraging textual reasoning to incorporate common sense. With a multi-round guidance and answer evolution framework, the proposed Code-as-Symbolic-Planner improves success rates by average 24.1\% over best baseline methods across seven typical TAMP tasks and three popular LLMs. Code-as-Symbolic-Planner shows strong effectiveness and generalizability across discrete and continuous environments, 2D/3D simulations and real-world settings, as well as single- and multi-robot tasks with diverse requirements. See our project website this https URL for prompts, videos, and code. 

---
# Using (Not so) Large Language Models for Generating Simulation Models in a Formal DSL -- A Study on Reaction Networks 

**Authors**: Justin N. Kreikemeyer, Miłosz Jankowski, Pia Wilsdorf, Adelinde M. Uhrmacher  

**Link**: [PDF](https://arxiv.org/pdf/2503.01675)  

**Abstract**: Formal languages are an integral part of modeling and simulation. They allow the distillation of knowledge into concise simulation models amenable to automatic execution, interpretation, and analysis. However, the arguably most humanly accessible means of expressing models is through natural language, which is not easily interpretable by computers. Here, we evaluate how a Large Language Model (LLM) might be used for formalizing natural language into simulation models. Existing studies only explored using very large LLMs, like the commercial GPT models, without fine-tuning model weights. To close this gap, we show how an open-weights, 7B-parameter Mistral model can be fine-tuned to translate natural language descriptions to reaction network models in a domain-specific language, offering a self-hostable, compute-, and memory efficient alternative. To this end, we develop a synthetic data generator to serve as the basis for fine-tuning and evaluation. Our quantitative evaluation shows that our fine-tuned Mistral model can recover the ground truth simulation model in up to 84.5% of cases. In addition, our small-scale user study demonstrates the model's practical potential for one-time generation as well as interactive modeling in various domains. While promising, in its current form, the fine-tuned small LLM cannot catch up with large LLMs. We conclude that higher-quality training data are required, and expect future small and open-source LLMs to offer new opportunities. 

---
# Lost in Moderation: How Commercial Content Moderation APIs Over- and Under-Moderate Group-Targeted Hate Speech and Linguistic Variations 

**Authors**: David Hartmann, Amin Oueslati, Dimitri Staufer, Lena Pohlmann, Simon Munzert, Hendrik Heuer  

**Link**: [PDF](https://arxiv.org/pdf/2503.01623)  

**Abstract**: Commercial content moderation APIs are marketed as scalable solutions to combat online hate speech. However, the reliance on these APIs risks both silencing legitimate speech, called over-moderation, and failing to protect online platforms from harmful speech, known as under-moderation. To assess such risks, this paper introduces a framework for auditing black-box NLP systems. Using the framework, we systematically evaluate five widely used commercial content moderation APIs. Analyzing five million queries based on four datasets, we find that APIs frequently rely on group identity terms, such as ``black'', to predict hate speech. While OpenAI's and Amazon's services perform slightly better, all providers under-moderate implicit hate speech, which uses codified messages, especially against LGBTQIA+ individuals. Simultaneously, they over-moderate counter-speech, reclaimed slurs and content related to Black, LGBTQIA+, Jewish, and Muslim people. We recommend that API providers offer better guidance on API implementation and threshold setting and more transparency on their APIs' limitations.
Warning: This paper contains offensive and hateful terms and concepts. We have chosen to reproduce these terms for reasons of transparency. 

---
# Advancing vision-language models in front-end development via data synthesis 

**Authors**: Tong Ge, Yashu Liu, Jieping Ye, Tianyi Li, Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01619)  

**Abstract**: Modern front-end (FE) development, especially when leveraging the unique features of frameworks like React and Vue, presents distinctive challenges. These include managing modular architectures, ensuring synchronization between data and visual outputs for declarative rendering, and adapting reusable components to various scenarios. Such complexities make it particularly difficult for state-of-the-art large vision-language models (VLMs) to generate accurate and functional code directly from design images. To address these challenges, we propose a reflective agentic workflow that synthesizes high-quality image-text data to capture the diverse characteristics of FE development. This workflow automates the extraction of self-contained\footnote{A \textbf{self-contained} code snippet is one that encapsulates all necessary logic, styling, and dependencies, ensuring it functions independently without requiring external imports or context.} code snippets from real-world projects, renders the corresponding visual outputs, and generates detailed descriptions that link design elements to functional code. To further expand the scope and utility of the synthesis, we introduce three data synthesis strategies: Evolution-based synthesis, which enables scalable and diverse dataset expansion; Waterfall-Model-based synthesis, which generates logically coherent code derived from system requirements; and Additive Development synthesis, which iteratively increases the complexity of human-authored components. We build a large vision-language model, Flame, trained on the synthesized datasets and demonstrate its effectiveness in generating React code via the $\text{pass}@k$ metric. Our results suggest that a code VLM trained to interpret images before code generation may achieve better performance. 

---
# EliteKV: Scalable KV Cache Compression via RoPE Frequency Selection and Joint Low-Rank Projection 

**Authors**: Yuhao Zhou, Sirui Song, Boyang Liu, Zhiheng Xi, Senjie Jin, Xiaoran Fan, Zhihao Zhang, Wei Li, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01586)  

**Abstract**: Rotary Position Embedding (RoPE) enables each attention head to capture multi-frequency information along the sequence dimension and is widely applied in foundation models. However, the nonlinearity introduced by RoPE complicates optimization of the key state in the Key-Value (KV) cache for RoPE-based attention. Existing KV cache compression methods typically store key state before rotation and apply the transformation during decoding, introducing additional computational overhead. This paper introduces EliteKV, a flexible modification framework for RoPE-based models supporting variable KV cache compression ratios. EliteKV first identifies the intrinsic frequency preference of each head using RoPElite, selectively restoring linearity to certain dimensions of key within attention computation. Building on this, joint low-rank compression of key and value enables partial cache sharing. Experimental results show that with minimal uptraining on only $0.6\%$ of the original training data, RoPE-based models achieve a $75\%$ reduction in KV cache size while preserving performance within a negligible margin. Furthermore, EliteKV consistently performs well across models of different scales within the same family. 

---
# None of the Above, Less of the Right: Parallel Patterns between Humans and LLMs on Multi-Choice Questions Answering 

**Authors**: Zhi Rui Tam, Cheng-Kuang Wu, Chieh-Yen Lin, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01550)  

**Abstract**: Multiple-choice exam questions with "None of the above" (NA) options have been extensively studied in educational testing, in which existing research suggests that they better assess true knowledge. However, their impact on Large Language Models (LLMs) evaluation remains underexplored. Through systematic experiments with 28 LLMs on the MMLU benchmark, we examine how NA options affect model performance and confidence calibration. Our analysis reveals that NA options, when used as the correct answer, lead to a consistent 30-50\% performance drop across models regardless of scale--suggesting that LLMs lack the meta-cognitive ability to systematically evaluate and reject all given options when none are correct. This degradation shows strong domain dependence, with minimal impact on mathematical reasoning (14.6\% drop) but severe effects on tasks requiring uncertainty handling like business ethics (48.1\% drop). Our results highlight important implications for benchmark design and raise questions about LLMs' ability to handle uncertainty in real-world applications. 

---
# Compositional Reasoning with Transformers, RNNs, and Chain of Thought 

**Authors**: Gilad Yehudai, Noah Amsel, Joan Bruna  

**Link**: [PDF](https://arxiv.org/pdf/2503.01544)  

**Abstract**: We study and compare the expressive power of transformers, RNNs, and transformers with chain of thought tokens on a simple and natural class of problems we term Compositional Reasoning Questions (CRQ). This family captures problems like evaluating Boolean formulas and multi-step word problems. Assuming standard hardness assumptions from circuit complexity and communication complexity, we prove that none of these three architectures is capable of solving CRQs unless some hyperparameter (depth, embedding dimension, and number of chain of thought tokens, respectively) grows with the size of the input. We also provide a construction for each architecture that solves CRQs. For transformers, our construction uses depth that is logarithmic in the problem size. For RNNs, logarithmic embedding dimension is necessary and sufficient, so long as the inputs are provided in a certain order. (Otherwise, a linear dimension is necessary). For transformers with chain of thought, our construction uses $n$ CoT tokens. These results show that, while CRQs are inherently hard, there are several different ways for language models to overcome this hardness. Even for a single class of problems, each architecture has strengths and weaknesses, and none is strictly better than the others. 

---
# Towards Widening The Distillation Bottleneck for Reasoning Models 

**Authors**: Huifeng Yin, Yu Zhao, Minghao Wu, Xuanfan Ni, Bo Zeng, Hao Wang, Tianqi Shi, Liangying Shao, Chenyang Lyu, Longyue Wang, Weihua Luo, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01461)  

**Abstract**: Large Reasoning Models(LRMs) such as OpenAI o1 and DeepSeek-R1 have shown remarkable reasoning capabilities by scaling test-time compute and generating long Chain-of-Thought(CoT). Distillation--post-training on LRMs-generated data--is a straightforward yet effective method to enhance the reasoning abilities of smaller models, but faces a critical bottleneck: we found that distilled long CoT data poses learning difficulty for small models and leads to the inheritance of biases (i.e. over-thinking) when using Supervised Fine-tuning(SFT) and Reinforcement Learning(RL) methods. To alleviate this bottleneck, we propose constructing tree-based CoT data from scratch via Monte Carlo Tree Search(MCTS). We then exploit a set of CoT-aware approaches, including Thoughts Length Balance, Fine-grained DPO, and Joint Post-training Objective, to enhance SFT and RL on the construted data. 

---
# AC-Lite : A Lightweight Image Captioning Model for Low-Resource Assamese Language 

**Authors**: Pankaj Choudhury, Yogesh Aggarwal, Prithwijit Guha, Sukumar Nandi  

**Link**: [PDF](https://arxiv.org/pdf/2503.01453)  

**Abstract**: Neural networks have significantly advanced AI applications, yet their real-world adoption remains constrained by high computational demands, hardware limitations, and accessibility challenges. In image captioning, many state-of-the-art models have achieved impressive performances while relying on resource-intensive architectures. This made them impractical for deployment on resource-constrained devices. This limitation is particularly noticeable for applications involving low-resource languages. We demonstrate the case of image captioning in Assamese language, where lack of effective, scalable systems can restrict the accessibility of AI-based solutions for native Assamese speakers. This work presents AC-Lite, a computationally efficient model for image captioning in low-resource Assamese language. AC-Lite reduces computational requirements by replacing computation-heavy visual feature extractors like FasterRCNN with lightweight ShuffleNetv2x1.5. Additionally, Gated Recurrent Units (GRUs) are used as the caption decoder to further reduce computational demands and model parameters. Furthermore, the integration of bilinear attention enhances the model's overall performance. AC-Lite can operate on edge devices, thereby eliminating the need for computation on remote servers. The proposed AC-Lite model achieves 82.3 CIDEr score on the COCO-AC dataset with 1.098 GFLOPs and 25.65M parameters. 

---
# From Hypothesis to Publication: A Comprehensive Survey of AI-Driven Research Support Systems 

**Authors**: Zekun Zhou, Xiaocheng Feng, Lei Huang, Xiachong Feng, Ziyun Song, Ruihan Chen, Liang Zhao, Weitao Ma, Yuxuan Gu, Baoxin Wang, Dayong Wu, Guoping Hu, Ting Liu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2503.01424)  

**Abstract**: Research is a fundamental process driving the advancement of human civilization, yet it demands substantial time and effort from researchers. In recent years, the rapid development of artificial intelligence (AI) technologies has inspired researchers to explore how AI can accelerate and enhance research. To monitor relevant advancements, this paper presents a systematic review of the progress in this domain. Specifically, we organize the relevant studies into three main categories: hypothesis formulation, hypothesis validation, and manuscript publication. Hypothesis formulation involves knowledge synthesis and hypothesis generation. Hypothesis validation includes the verification of scientific claims, theorem proving, and experiment validation. Manuscript publication encompasses manuscript writing and the peer review process. Furthermore, we identify and discuss the current challenges faced in these areas, as well as potential future directions for research. Finally, we also offer a comprehensive overview of existing benchmarks and tools across various domains that support the integration of AI into the research process. We hope this paper serves as an introduction for beginners and fosters future research. Resources have been made publicly available at this https URL. 

---
# Generalizable Prompt Learning of CLIP: A Brief Overview 

**Authors**: Fangming Cui, Yonggang Zhang, Xuan Wang, Xule Wang, Liang Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.01263)  

**Abstract**: Existing vision-language models (VLMs) such as CLIP have showcased an impressive capability to generalize well across various downstream tasks. These models leverage the synergy between visual and textual information, enabling them to understand and reason about the content present in images and text in a unified manner. This article provides a brief overview of CLIP based on few-shot prompt learning, including experimental data and technical characteristics of some methods. The purpose of this review is to provide a reference for researchers who have just started their research in generalizable prompting of CLIP through few-shot training for classification across 15 datasets and also to facilitate the integration of this field by researchers in other downstream tasks. 

---
# Retrieval-Augmented Perception: High-Resolution Image Perception Meets Visual RAG 

**Authors**: Wenbin Wang, Yongcheng Jing, Liang Ding, Yingjie Wang, Li Shen, Yong Luo, Bo Du, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.01222)  

**Abstract**: High-resolution (HR) image perception remains a key challenge in multimodal large language models (MLLMs). To overcome the limitations of existing methods, this paper shifts away from prior dedicated heuristic approaches and revisits the most fundamental idea to HR perception by enhancing the long-context capability of MLLMs, driven by recent advances in long-context techniques like retrieval-augmented generation (RAG) for general LLMs. Towards this end, this paper presents the first study exploring the use of RAG to address HR perception challenges. Specifically, we propose Retrieval-Augmented Perception (RAP), a training-free framework that retrieves and fuses relevant image crops while preserving spatial context using the proposed Spatial-Awareness Layout. To accommodate different tasks, the proposed Retrieved-Exploration Search (RE-Search) dynamically selects the optimal number of crops based on model confidence and retrieval scores. Experimental results on HR benchmarks demonstrate the significant effectiveness of RAP, with LLaVA-v1.5-13B achieving a 43% improvement on $V^*$ Bench and 19% on HR-Bench. 

---
# Watch Out Your Album! On the Inadvertent Privacy Memorization in Multi-Modal Large Language Models 

**Authors**: Tianjie Ju, Yi Hua, Hao Fei, Zhenyu Shao, Yubin Zheng, Haodong Zhao, Mong-Li Lee, Wynne Hsu, Zhuosheng Zhang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01208)  

**Abstract**: Multi-Modal Large Language Models (MLLMs) have exhibited remarkable performance on various vision-language tasks such as Visual Question Answering (VQA). Despite accumulating evidence of privacy concerns associated with task-relevant content, it remains unclear whether MLLMs inadvertently memorize private content that is entirely irrelevant to the training tasks. In this paper, we investigate how randomly generated task-irrelevant private content can become spuriously correlated with downstream objectives due to partial mini-batch training dynamics, thus causing inadvertent memorization. Concretely, we randomly generate task-irrelevant watermarks into VQA fine-tuning images at varying probabilities and propose a novel probing framework to determine whether MLLMs have inadvertently encoded such content. Our experiments reveal that MLLMs exhibit notably different training behaviors in partial mini-batch settings with task-irrelevant watermarks embedded. Furthermore, through layer-wise probing, we demonstrate that MLLMs trigger distinct representational patterns when encountering previously seen task-irrelevant knowledge, even if this knowledge does not influence their output during prompting. Our code is available at this https URL. 

---
# Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers 

**Authors**: Rin Ashizawa, Yoichi Hirose, Nozomu Yoshinari, Kento Uchida, Shinichi Shirakawa  

**Link**: [PDF](https://arxiv.org/pdf/2503.01163)  

**Abstract**: Prompt optimization aims to search for effective prompts that enhance the performance of large language models (LLMs). Although existing prompt optimization methods have discovered effective prompts, they often differ from sophisticated prompts carefully designed by human experts. Prompt design strategies, representing best practices for improving prompt performance, can be key to improving prompt optimization. Recently, a method termed the Autonomous Prompt Engineering Toolbox (APET) has incorporated various prompt design strategies into the prompt optimization process. In APET, the LLM is needed to implicitly select and apply the appropriate strategies because prompt design strategies can have negative effects. This implicit selection may be suboptimal due to the limited optimization capabilities of LLMs. This paper introduces Optimizing Prompts with sTrategy Selection (OPTS), which implements explicit selection mechanisms for prompt design. We propose three mechanisms, including a Thompson sampling-based approach, and integrate them into EvoPrompt, a well-known prompt optimizer. Experiments optimizing prompts for two LLMs, Llama-3-8B-Instruct and GPT-4o mini, were conducted using BIG-Bench Hard. Our results show that the selection of prompt design strategies improves the performance of EvoPrompt, and the Thompson sampling-based mechanism achieves the best overall results. Our experimental code is provided at this https URL . 

---
# SolBench: A Dataset and Benchmark for Evaluating Functional Correctness in Solidity Code Completion and Repair 

**Authors**: Zaoyu Chen, Haoran Qin, Nuo Chen, Xiangyu Zhao, Lei Xue, Xiapu Luo, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01098)  

**Abstract**: Smart contracts are crucial programs on blockchains, and their immutability post-deployment makes functional correctness vital. Despite progress in code completion models, benchmarks for Solidity, the primary smart contract language, are lacking. Existing metrics like BLEU do not adequately assess the functional correctness of generated smart contracts. To fill this gap, we introduce SolBench, a benchmark for evaluating the functional correctness of Solidity smart contracts generated by code completion models. SolBench includes 4,178 functions from 1,155 Ethereum-deployed contracts. Testing advanced models revealed challenges in generating correct code without context, as Solidity functions rely on context-defined variables and interfaces. To address this, we propose a Retrieval-Augmented Code Repair framework. In this framework, an executor verifies functional correctness, and if necessary, an LLM repairs the code using retrieved snippets informed by executor traces. We conduct a comprehensive evaluation of both closed-source and open-source LLMs across various model sizes and series to assess their performance in smart contract completion. The results show that code repair and retrieval techniques effectively enhance the correctness of smart contract completion while reducing computational costs. 

---
# Alchemist: Towards the Design of Efficient Online Continual Learning System 

**Authors**: Yuyang Huang, Yuhan Liu, Haryadi S. Gunawi, Beibin Li, Changho Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01066)  

**Abstract**: Continual learning has emerged as a promising solution to refine models incrementally by leveraging user feedback, thereby enhancing model performance in applications like code completion, personal assistants, and chat interfaces. In particular, online continual learning - iteratively training the model with small batches of user feedback - has demonstrated notable performance improvements. However, the existing practice of segregating training and serving processes forces the online trainer to recompute the intermediate results already done during serving. Such redundant computations can account for 30%-42% of total training time.
In this paper, we propose Alchemist, to the best of our knowledge, the first online continual learning system that efficiently reuses intermediate results computed during serving to reduce redundant computation with minimal impact on the serving latency or capacity. Alchemist introduces two key techniques: (1) minimal activations recording and saving during serving, where activations are recorded and saved only during the prefill phase to minimize overhead; and (2) offloading of serving activations, which dynamically manages GPU memory by freeing activations in the forward order, while reloading them in the backward order during the backward pass. Evaluations with the ShareGPT dataset show that compared with a separate training cluster, Alchemist significantly increases training throughput by up to 1.72x, reduces up to 47% memory usage during training, and supports up to 2x more training tokens - all while maintaining negligible impact on serving latency. 

---
# Variance reduction in output from generative AI 

**Authors**: Yu Xie, Yueqi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2503.01033)  

**Abstract**: Generative AI models, such as ChatGPT, will increasingly replace humans in producing output for a variety of important tasks. While much prior work has mostly focused on the improvement in the average performance of generative AI models relative to humans' performance, much less attention has been paid to the significant reduction of variance in output produced by generative AI models. In this Perspective, we demonstrate that generative AI models are inherently prone to the phenomenon of "regression toward the mean" whereby variance in output tends to shrink relative to that in real-world distributions. We discuss potential social implications of this phenomenon across three levels-societal, group, and individual-and two dimensions-material and non-material. Finally, we discuss interventions to mitigate negative effects, considering the roles of both service providers and users. Overall, this Perspective aims to raise awareness of the importance of output variance in generative AI and to foster collaborative efforts to meet the challenges posed by the reduction of variance in output generated by AI models. 

---
# A Semantic Search Pipeline for Causality-driven Adhoc Information Retrieval 

**Authors**: Dhairya Dalal, Sharmi Dev Gupta, Bentolhoda Binaei  

**Link**: [PDF](https://arxiv.org/pdf/2503.01003)  

**Abstract**: We present a unsupervised semantic search pipeline for the Causality-driven Adhoc Information Retrieval (CAIR-2021) shared task. The CAIR shared task expands traditional information retrieval to support the retrieval of documents containing the likely causes of a query event. A successful system must be able to distinguish between topical documents and documents containing causal descriptions of events that are causally related to the query event. Our approach involves aggregating results from multiple query strategies over a semantic and lexical index. The proposed approach leads the CAIR-2021 leaderboard and outperformed both traditional IR and pure semantic embedding-based approaches. 

---
# Evidence of conceptual mastery in the application of rules by Large Language Models 

**Authors**: José Luiz Nunes, Guilherme FCF Almeida, Brian Flanagan  

**Link**: [PDF](https://arxiv.org/pdf/2503.00992)  

**Abstract**: In this paper we leverage psychological methods to investigate LLMs' conceptual mastery in applying rules. We introduce a novel procedure to match the diversity of thought generated by LLMs to that observed in a human sample. We then conducted two experiments comparing rule-based decision-making in humans and LLMs. Study 1 found that all investigated LLMs replicated human patterns regardless of whether they are prompted with scenarios created before or after their training cut-off. Moreover, we found unanticipated differences between the two sets of scenarios among humans. Surprisingly, even these differences were replicated in LLM responses. Study 2 turned to a contextual feature of human rule application: under forced time delay, human samples rely more heavily on a rule's text than on other considerations such as a rule's purpose.. Our results revealed that some models (Gemini Pro and Claude 3) responded in a human-like manner to a prompt describing either forced delay or time pressure, while others (GPT-4o and Llama 3.2 90b) did not. We argue that the evidence gathered suggests that LLMs have mastery over the concept of rule, with implications for both legal decision making and philosophical inquiry. 

---
# UniWav: Towards Unified Pre-training for Speech Representation Learning and Generation 

**Authors**: Alexander H. Liu, Sang-gil Lee, Chao-Han Huck Yang, Yuan Gong, Yu-Chiang Frank Wang, James R. Glass, Rafael Valle, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2503.00733)  

**Abstract**: Pre-training and representation learning have been playing an increasingly important role in modern speech processing. Nevertheless, different applications have been relying on different foundation models, since predominant pre-training techniques are either designed for discriminative tasks or generative tasks. In this work, we make the first attempt at building a unified pre-training framework for both types of tasks in speech. We show that with the appropriate design choices for pre-training, one can jointly learn a representation encoder and generative audio decoder that can be applied to both types of tasks. We propose UniWav, an encoder-decoder framework designed to unify pre-training representation learning and generative tasks. On speech recognition, text-to-speech, and speech tokenization, UniWav achieves comparable performance to different existing foundation models, each trained on a specific task. Our findings suggest that a single general-purpose foundation model for speech can be built to replace different foundation models, reducing the overhead and cost of pre-training. 

---
# Causal Inference on Outcomes Learned from Text 

**Authors**: Iman Modarressi, Jann Spiess, Amar Venugopal  

**Link**: [PDF](https://arxiv.org/pdf/2503.00725)  

**Abstract**: We propose a machine-learning tool that yields causal inference on text in randomized trials. Based on a simple econometric framework in which text may capture outcomes of interest, our procedure addresses three questions: First, is the text affected by the treatment? Second, which outcomes is the effect on? And third, how complete is our description of causal effects? To answer all three questions, our approach uses large language models (LLMs) that suggest systematic differences across two groups of text documents and then provides valid inference based on costly validation. Specifically, we highlight the need for sample splitting to allow for statistical validation of LLM outputs, as well as the need for human labeling to validate substantive claims about how documents differ across groups. We illustrate the tool in a proof-of-concept application using abstracts of academic manuscripts. 

---
# Towards hyperparameter-free optimization with differential privacy 

**Authors**: Zhiqi Bu, Ruixuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00703)  

**Abstract**: Differential privacy (DP) is a privacy-preserving paradigm that protects the training data when training deep learning models. Critically, the performance of models is determined by the training hyperparameters, especially those of the learning rate schedule, thus requiring fine-grained hyperparameter tuning on the data. In practice, it is common to tune the learning rate hyperparameters through the grid search that (1) is computationally expensive as multiple runs are needed, and (2) increases the risk of data leakage as the selection of hyperparameters is data-dependent. In this work, we adapt the automatic learning rate schedule to DP optimization for any models and optimizers, so as to significantly mitigate or even eliminate the cost of hyperparameter tuning when applied together with automatic per-sample gradient clipping. Our hyperparameter-free DP optimization is almost as computationally efficient as the standard non-DP optimization, and achieves state-of-the-art DP performance on various language and vision tasks. 

---
# How Diversely Can Language Models Solve Problems? Exploring the Algorithmic Diversity of Model-Generated Code 

**Authors**: Seonghyeon Lee, Heejae Chon, Joonwon Jang, Dongha Lee, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00691)  

**Abstract**: Language models (LMs) have exhibited impressive abilities in generating code from natural language requirements. In this work, we highlight the diversity of code generated by LMs as a critical criterion for evaluating their code generation capabilities. There is a lack of studies focused on assessing the diversity of generated code, which overlooks its importance in code LMs. Therefore, we propose a systematic approach to evaluate code diversity, introducing various metrics with inter-code similarity. Specifically, we introduce code clustering methods that leverages LMs' capabilities in code understanding and reasoning, resulting in a set of metrics that represent the number of algorithms in model-generated solutions. We extensively investigate the property of model-generated solutions by contrasting them with human-written ones and quantifying the impact of various factors on code diversity: model size, temperature, instruction tuning, and problem complexity. Our analysis demonstrates that model-generated solutions exhibit low algorithmic diversity, which was neglected by the research community. Moreover, we explore methods to increase code diversity by combining solutions from different models and increasing sampling temperatures. Our findings highlight that code diversity can be enhanced with the help of heterogeneous models and setting temperature beyond 1.0 that has not been fully explored due to the functional correctness degradation. To facilitate our research direction, we publicly share our code and datasets through open-source repositories. 

---
# Statistical Mechanics of Semantic Compression 

**Authors**: Tankut Can  

**Link**: [PDF](https://arxiv.org/pdf/2503.00612)  

**Abstract**: The basic problem of semantic compression is to minimize the length of a message while preserving its meaning. This differs from classical notions of compression in that the distortion is not measured directly at the level of bits, but rather in an abstract semantic space. In order to make this precise, we take inspiration from cognitive neuroscience and machine learning and model semantic space as a continuous Euclidean vector space. In such a space, stimuli like speech, images, or even ideas, are mapped to high-dimensional real vectors, and the location of these embeddings determines their meaning relative to other embeddings. This suggests that a natural metric for semantic similarity is just the Euclidean distance, which is what we use in this work. We map the optimization problem of determining the minimal-length, meaning-preserving message to a spin glass Hamiltonian and solve the resulting statistical mechanics problem using replica theory. We map out the replica symmetric phase diagram, identifying distinct phases of semantic compression: a first-order transition occurs between lossy and lossless compression, whereas a continuous crossover is seen from extractive to abstractive compression. We conclude by showing numerical simulations of compressions obtained by simulated annealing and greedy algorithms, and argue that while the problem of finding a meaning-preserving compression is computationally hard in the worst case, there exist efficient algorithms which achieve near optimal performance in the typical case. 

---
# Semantic Integrity Constraints: Declarative Guardrails for AI-Augmented Data Processing Systems 

**Authors**: Alexander W. Lee, Justin Chan, Michael Fu, Nicolas Kim, Akshay Mehta, Deepti Raghavan, Ugur Cetintemel  

**Link**: [PDF](https://arxiv.org/pdf/2503.00600)  

**Abstract**: The emergence of AI-augmented Data Processing Systems (DPSs) has introduced powerful semantic operators that extend traditional data management capabilities with LLM-based processing. However, these systems face fundamental reliability (a.k.a. trust) challenges, as LLMs can generate erroneous outputs, limiting their adoption in critical domains. Existing approaches to LLM constraints--ranging from user-defined functions to constrained decoding--are fragmented, imperative, and lack semantics-aware integration into query execution. To address this gap, we introduce Semantic Integrity Constraints (SICs), a novel declarative abstraction that extends traditional database integrity constraints to govern and optimize semantic operators within DPSs. SICs integrate seamlessly into the relational model, allowing users to specify common classes of constraints (e.g., grounding and soundness) while enabling query-aware enforcement and optimization strategies.
In this paper, we present the core design of SICs, describe their formal integration into query execution, and detail our conception of grounding constraints, a key SIC class that ensures factual consistency of generated outputs. In addition, we explore novel enforcement mechanisms, combining proactive (constrained decoding) and reactive (validation and recovery) techniques to optimize efficiency and reliability. Our work establishes SICs as a foundational framework for trustworthy, high-performance AI-augmented data processing, paving the way for future research in constraint-driven optimizations, adaptive enforcement, and enterprise-scale deployments. 

---
# Instructor-Worker Large Language Model System for Policy Recommendation: a Case Study on Air Quality Analysis of the January 2025 Los Angeles Wildfires 

**Authors**: Kyle Gao, Dening Lu, Liangzhi Li, Nan Chen, Hongjie He, Linlin Xu, Jonathan Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00566)  

**Abstract**: The Los Angeles wildfires of January 2025 caused more than 250 billion dollars in damage and lasted for nearly an entire month before containment. Following our previous work, the Digital Twin Building, we modify and leverage the multi-agent large language model framework as well as the cloud-mapping integration to study the air quality during the Los Angeles wildfires. Recent advances in large language models have allowed for out-of-the-box automated large-scale data analysis. We use a multi-agent large language system comprised of an Instructor agent and Worker agents. Upon receiving the users' instructions, the Instructor agent retrieves the data from the cloud platform and produces instruction prompts to the Worker agents. The Worker agents then analyze the data and provide summaries. The summaries are finally input back into the Instructor agent, which then provides the final data analysis. We test this system's capability for data-based policy recommendation by assessing our Instructor-Worker LLM system's health recommendations based on air quality during the Los Angeles wildfires. 

---
# Qilin: A Multimodal Information Retrieval Dataset with APP-level User Sessions 

**Authors**: Jia Chen, Qian Dong, Haitao Li, Xiaohui He, Yan Gao, Shaosheng Cao, Yi Wu, Ping Yang, Chen Xu, Yao Hu, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00501)  

**Abstract**: User-generated content (UGC) communities, especially those featuring multimodal content, improve user experiences by integrating visual and textual information into results (or items). The challenge of improving user experiences in complex systems with search and recommendation (S\&R) services has drawn significant attention from both academia and industry these years. However, the lack of high-quality datasets has limited the research progress on multimodal S\&R. To address the growing need for developing better S\&R services, we present a novel multimodal information retrieval dataset in this paper, namely Qilin. The dataset is collected from Xiaohongshu, a popular social platform with over 300 million monthly active users and an average search penetration rate of over 70\%. In contrast to existing datasets, \textsf{Qilin} offers a comprehensive collection of user sessions with heterogeneous results like image-text notes, video notes, commercial notes, and direct answers, facilitating the development of advanced multimodal neural retrieval models across diverse task settings. To better model user satisfaction and support the analysis of heterogeneous user behaviors, we also collect extensive APP-level contextual signals and genuine user feedback. Notably, Qilin contains user-favored answers and their referred results for search requests triggering the Deep Query Answering (DQA) module. This allows not only the training \& evaluation of a Retrieval-augmented Generation (RAG) pipeline, but also the exploration of how such a module would affect users' search behavior. Through comprehensive analysis and experiments, we provide interesting findings and insights for further improving S\&R systems. We hope that \textsf{Qilin} will significantly contribute to the advancement of multimodal content platforms with S\&R services in the future. 

---
# LLaSE-G1: Incentivizing Generalization Capability for LLaMA-based Speech Enhancement 

**Authors**: Boyi Kang, Xinfa Zhu, Zihan Zhang, Zhen Ye, Mingshuai Liu, Ziqian Wang, Yike Zhu, Guobin Ma, Jun Chen, Longshuai Xiao, Chao Weng, Wei Xue, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2503.00493)  

**Abstract**: Recent advancements in language models (LMs) have demonstrated strong capabilities in semantic understanding and contextual modeling, which have flourished in generative speech enhancement (SE). However, many LM-based SE approaches primarily focus on semantic information, often neglecting the critical role of acoustic information, which leads to acoustic inconsistency after enhancement and limited generalization across diverse SE tasks. In this paper, we introduce LLaSE-G1, a LLaMA-based language model that incentivizes generalization capabilities for speech enhancement. LLaSE-G1 offers the following key contributions: First, to mitigate acoustic inconsistency, LLaSE-G1 employs continuous representations from WavLM as input and predicts speech tokens from X-Codec2, maximizing acoustic preservation. Second, to promote generalization capability, LLaSE-G1 introduces dual-channel inputs and outputs, unifying multiple SE tasks without requiring task-specific IDs. Third, LLaSE-G1 outperforms prior task-specific discriminative and generative SE models, demonstrating scaling effects at test time and emerging capabilities for unseen SE tasks. Additionally, we release our code and models to support further research in this area. 

---
# Reducing Large Language Model Safety Risks in Women's Health using Semantic Entropy 

**Authors**: Jahan C. Penny-Dimri, Magdalena Bachmann, William R. Cooke, Sam Mathewlynn, Samuel Dockree, John Tolladay, Jannik Kossen, Lin Li, Yarin Gal, Gabriel Davis Jones  

**Link**: [PDF](https://arxiv.org/pdf/2503.00269)  

**Abstract**: Large language models (LLMs) hold substantial promise for clinical decision support. However, their widespread adoption in medicine, particularly in healthcare, is hindered by their propensity to generate false or misleading outputs, known as hallucinations. In high-stakes domains such as women's health (obstetrics & gynaecology), where errors in clinical reasoning can have profound consequences for maternal and neonatal outcomes, ensuring the reliability of AI-generated responses is critical. Traditional methods for quantifying uncertainty, such as perplexity, fail to capture meaning-level inconsistencies that lead to misinformation. Here, we evaluate semantic entropy (SE), a novel uncertainty metric that assesses meaning-level variation, to detect hallucinations in AI-generated medical content. Using a clinically validated dataset derived from UK RCOG MRCOG examinations, we compared SE with perplexity in identifying uncertain responses. SE demonstrated superior performance, achieving an AUROC of 0.76 (95% CI: 0.75-0.78), compared to 0.62 (0.60-0.65) for perplexity. Clinical expert validation further confirmed its effectiveness, with SE achieving near-perfect uncertainty discrimination (AUROC: 0.97). While semantic clustering was successful in only 30% of cases, SE remains a valuable tool for improving AI safety in women's health. These findings suggest that SE could enable more reliable AI integration into clinical practice, particularly in resource-limited settings where LLMs could augment care. This study highlights the potential of SE as a key safeguard in the responsible deployment of AI-driven tools in women's health, leading to safer and more effective digital health interventions. 

---
# CoSMoEs: Compact Sparse Mixture of Experts 

**Authors**: Patrick Huber, Akshat Shrivastava, Ernie Chang, Chinnadhurai Sankar, Ahmed Aly, Adithya Sagar  

**Link**: [PDF](https://arxiv.org/pdf/2503.00245)  

**Abstract**: Sparse Mixture of Expert (MoE) models are popular foundational architectures at large scale, however, under-explored at smaller sizes. Here, we show how to enable Compact Sparse Mixture of Experts (CoSMoEs) for on-device inference. Specifically, we tackle the three main on-device dimensions: Quality, Memory and Latency. Along the quality axis, we show that in a fair evaluation (removing confounding factors) MoE architectures outperform FLOP-aligned dense models at on-device scale. We introduce weight-decomposed experts, further improving the MoE model performance. Regarding model memory and latency, we significantly improve model offloading efficiency and, in turn, reduce model inference latency. 

---
# Zero-Shot and Efficient Clarification Need Prediction in Conversational Search 

**Authors**: Lili Lu, Chuan Meng, Federico Ravenda, Mohammad Aliannejadi, Fabio Crestani  

**Link**: [PDF](https://arxiv.org/pdf/2503.00179)  

**Abstract**: Clarification need prediction (CNP) is a key task in conversational search, aiming to predict whether to ask a clarifying question or give an answer to the current user query. However, current research on CNP suffers from the issues of limited CNP training data and low efficiency. In this paper, we propose a zero-shot and efficient CNP framework (Zef-CNP), in which we first prompt large language models (LLMs) in a zero-shot manner to generate two sets of synthetic queries: ambiguous and specific (unambiguous) queries. We then use the generated queries to train efficient CNP models. Zef-CNP eliminates the need for human-annotated clarification-need labels during training and avoids the use of LLMs with high query latency at query time. To further improve the generation quality of synthetic queries, we devise a topic-, information-need-, and query-aware chain-of-thought (CoT) prompting strategy (TIQ-CoT). Moreover, we enhance TIQ-CoT with counterfactual query generation (CoQu), which guides LLMs first to generate a specific/ambiguous query and then sequentially generate its corresponding ambiguous/specific query. Experimental results show that Zef-CNP achieves superior CNP effectiveness and efficiency compared with zero- and few-shot LLM-based CNP predictors. 

---
# PreMind: Multi-Agent Video Understanding for Advanced Indexing of Presentation-style Videos 

**Authors**: Kangda Wei, Zhengyu Zhou, Bingqing Wang, Jun Araki, Lukas Lange, Ruihong Huang, Zhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.00162)  

**Abstract**: In recent years, online lecture videos have become an increasingly popular resource for acquiring new knowledge. Systems capable of effectively understanding/indexing lecture videos are thus highly desirable, enabling downstream tasks like question answering to help users efficiently locate specific information within videos. This work proposes PreMind, a novel multi-agent multimodal framework that leverages various large models for advanced understanding/indexing of presentation-style videos. PreMind first segments videos into slide-presentation segments using a Vision-Language Model (VLM) to enhance modern shot-detection techniques. Each segment is then analyzed to generate multimodal indexes through three key steps: (1) extracting slide visual content, (2) transcribing speech narratives, and (3) consolidating these visual and speech contents into an integrated understanding. Three innovative mechanisms are also proposed to improve performance: leveraging prior lecture knowledge to refine visual understanding, detecting/correcting speech transcription errors using a VLM, and utilizing a critic agent for dynamic iterative self-reflection in vision analysis. Compared to traditional video indexing methods, PreMind captures rich, reliable multimodal information, allowing users to search for details like abbreviations shown only on slides. Systematic evaluations on the public LPM dataset and an internal enterprise dataset are conducted to validate PreMind's effectiveness, supported by detailed analyses. 

---
# InspireMusic: Integrating Super Resolution and Large Language Model for High-Fidelity Long-Form Music Generation 

**Authors**: Chong Zhang, Yukun Ma, Qian Chen, Wen Wang, Shengkui Zhao, Zexu Pan, Hao Wang, Chongjia Ni, Trung Hieu Nguyen, Kun Zhou, Yidi Jiang, Chaohong Tan, Zhifu Gao, Zhihao Du, Bin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.00084)  

**Abstract**: We introduce InspireMusic, a framework integrated super resolution and large language model for high-fidelity long-form music generation. A unified framework generates high-fidelity music, songs, and audio, which incorporates an autoregressive transformer with a super-resolution flow-matching model. This framework enables the controllable generation of high-fidelity long-form music at a higher sampling rate from both text and audio prompts. Our model differs from previous approaches, as we utilize an audio tokenizer with one codebook that contains richer semantic information, thereby reducing training costs and enhancing efficiency. This combination enables us to achieve high-quality audio generation with long-form coherence of up to $8$ minutes. Then, an autoregressive transformer model based on Qwen 2.5 predicts audio tokens. Next, we employ a super-resolution flow-matching model to generate high-sampling rate audio with fine-grained details learned from an acoustic codec model. Comprehensive experiments show that the InspireMusic-1.5B-Long model has a comparable performance to recent top-tier open-source systems, including MusicGen and Stable Audio 2.0, on subjective and objective evaluations. The code and pre-trained models are released at this https URL. 

---
# I see what you mean: Co-Speech Gestures for Reference Resolution in Multimodal Dialogue 

**Authors**: Esam Ghaleb, Bulat Khaertdinov, Aslı Özyürek, Raquel Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2503.00071)  

**Abstract**: In face-to-face interaction, we use multiple modalities, including speech and gestures, to communicate information and resolve references to objects. However, how representational co-speech gestures refer to objects remains understudied from a computational perspective. In this work, we address this gap by introducing a multimodal reference resolution task centred on representational gestures, while simultaneously tackling the challenge of learning robust gesture embeddings. We propose a self-supervised pre-training approach to gesture representation learning that grounds body movements in spoken language. Our experiments show that the learned embeddings align with expert annotations and have significant predictive power. Moreover, reference resolution accuracy further improves when (1) using multimodal gesture representations, even when speech is unavailable at inference time, and (2) leveraging dialogue history. Overall, our findings highlight the complementary roles of gesture and speech in reference resolution, offering a step towards more naturalistic models of human-machine interaction. 

---
# Societal Alignment Frameworks Can Improve LLM Alignment 

**Authors**: Karolina Stańczak, Nicholas Meade, Mehar Bhatia, Hattie Zhou, Konstantin Böttinger, Jeremy Barnes, Jason Stanley, Jessica Montgomery, Richard Zemel, Nicolas Papernot, Nicolas Chapados, Denis Therien, Timothy P. Lillicrap, Ana Marasović, Sylvie Delacroix, Gillian K. Hadfield, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2503.00069)  

**Abstract**: Recent progress in large language models (LLMs) has focused on producing responses that meet human expectations and align with shared values - a process coined alignment. However, aligning LLMs remains challenging due to the inherent disconnect between the complexity of human values and the narrow nature of the technological approaches designed to address them. Current alignment methods often lead to misspecified objectives, reflecting the broader issue of incomplete contracts, the impracticality of specifying a contract between a model developer, and the model that accounts for every scenario in LLM alignment. In this paper, we argue that improving LLM alignment requires incorporating insights from societal alignment frameworks, including social, economic, and contractual alignment, and discuss potential solutions drawn from these domains. Given the role of uncertainty within societal alignment frameworks, we then investigate how it manifests in LLM alignment. We end our discussion by offering an alternative view on LLM alignment, framing the underspecified nature of its objectives as an opportunity rather than perfect their specification. Beyond technical improvements in LLM alignment, we discuss the need for participatory alignment interface designs. 

---
# Improved YOLOv12 with LLM-Generated Synthetic Data for Enhanced Apple Detection and Benchmarking Against YOLOv11 and YOLOv10 

**Authors**: Ranjan Sapkota, Manoj Karkee  

**Link**: [PDF](https://arxiv.org/pdf/2503.00057)  

**Abstract**: This study evaluated the performance of the YOLOv12 object detection model, and compared against YOLOv11 and YOLOv10 for apple detection in commercial orchards using synthetic images generated by Large Language Models (LLMs). The YOLOv12n configuration excelled, achieving the highest precision at 0.916, the highest recall at 0.969, and the highest mean Average Precision (mAP@50) at 0.978. In comparison, the YOLOv11 series was led by YOLO11x, which recorded the highest precision at 0.857, recall at 0.85, and mAP@50 at 0.91. For the YOLOv10 series, YOLOv10b and YOLOv10l tied for the highest precision at 0.85, with YOLOv10n achieving the highest recall at 0.8 and mAP@50 at 0.89. The study also highlighted efficiency in processing speeds, where YOLOv11n reported the lowest inference time at 4.7 ms, compared to YOLOv12n's 5.6 ms and YOLOv10n's 5.9 ms. Although YOLOv12 is new in more accurate than YOLOv11, and YOLOv10, the YOLO11n still stays the fastest YOLO algorithm among YOLOv10, YOLOv11 and YOLOv12. These findings demonstrated that YOLOv12, when trained on high-quality LLM-generated datasets, not only surpassed its predecessors in key performance metrics but also offered a cost-effective solution by reducing the need for extensive manual data collection in the field. 

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
# DeepSolution: Boosting Complex Engineering Solution Design via Tree-based Exploration and Bi-point Thinking 

**Authors**: Zhuoqun Li, Haiyang Yu, Xuanang Chen, Hongyu Lin, Yaojie Lu, Fei Huang, Xianpei Han, Yongbin Li, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.20730)  

**Abstract**: Designing solutions for complex engineering challenges is crucial in human production activities. However, previous research in the retrieval-augmented generation (RAG) field has not sufficiently addressed tasks related to the design of complex engineering solutions. To fill this gap, we introduce a new benchmark, SolutionBench, to evaluate a system's ability to generate complete and feasible solutions for engineering problems with multiple complex constraints. To further advance the design of complex engineering solutions, we propose a novel system, SolutionRAG, that leverages the tree-based exploration and bi-point thinking mechanism to generate reliable solutions. Extensive experimental results demonstrate that SolutionRAG achieves state-of-the-art (SOTA) performance on the SolutionBench, highlighting its potential to enhance the automation and reliability of complex engineering solution design in real-world applications. 

---
# Protecting Users From Themselves: Safeguarding Contextual Privacy in Interactions with Conversational Agents 

**Authors**: Ivoline Ngong, Swanand Kadhe, Hao Wang, Keerthiram Murugesan, Justin D. Weisz, Amit Dhurandhar, Karthikeyan Natesan Ramamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2502.18509)  

**Abstract**: Conversational agents are increasingly woven into individuals' personal lives, yet users often underestimate the privacy risks involved. The moment users share information with these agents (e.g., LLMs), their private information becomes vulnerable to exposure. In this paper, we characterize the notion of contextual privacy for user interactions with LLMs. It aims to minimize privacy risks by ensuring that users (sender) disclose only information that is both relevant and necessary for achieving their intended goals when interacting with LLMs (untrusted receivers). Through a formative design user study, we observe how even "privacy-conscious" users inadvertently reveal sensitive information through indirect disclosures. Based on insights from this study, we propose a locally-deployable framework that operates between users and LLMs, and identifies and reformulates out-of-context information in user prompts. Our evaluation using examples from ShareGPT shows that lightweight models can effectively implement this framework, achieving strong gains in contextual privacy while preserving the user's intended interaction goals through different approaches to classify information relevant to the intended goals. 

---
