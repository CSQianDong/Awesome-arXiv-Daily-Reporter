# ZeroSearch: Incentivize the Search Capability of LLMs without Searching 

**Authors**: Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan Hou, Yong Jiang, Pengjun Xie, Fei Huang, Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04588)  

**Abstract**: Effective information searching is essential for enhancing the reasoning and generation capabilities of large language models (LLMs). Recent research has explored using reinforcement learning (RL) to improve LLMs' search capabilities by interacting with live search engines in real-world environments. While these approaches show promising results, they face two major challenges: (1) Uncontrolled Document Quality: The quality of documents returned by search engines is often unpredictable, introducing noise and instability into the training process. (2) Prohibitively High API Costs: RL training requires frequent rollouts, potentially involving hundreds of thousands of search requests, which incur substantial API expenses and severely constrain scalability. To address these challenges, we introduce ZeroSearch, a reinforcement learning framework that incentivizes the search capabilities of LLMs without interacting with real search engines. Our approach begins with lightweight supervised fine-tuning to transform the LLM into a retrieval module capable of generating both relevant and noisy documents in response to a query. During RL training, we employ a curriculum-based rollout strategy that incrementally degrades the quality of generated documents, progressively eliciting the model's reasoning ability by exposing it to increasingly challenging retrieval scenarios. Extensive experiments demonstrate that ZeroSearch effectively incentivizes the search capabilities of LLMs using a 3B LLM as the retrieval module. Remarkably, a 7B retrieval module achieves comparable performance to the real search engine, while a 14B retrieval module even surpasses it. Furthermore, it generalizes well across both base and instruction-tuned models of various parameter sizes and is compatible with a wide range of RL algorithms. 

---
# Overcoming Data Scarcity in Generative Language Modelling for Low-Resource Languages: A Systematic Review 

**Authors**: Josh McGiff, Nikola S. Nikolov  

**Link**: [PDF](https://arxiv.org/pdf/2505.04531)  

**Abstract**: Generative language modelling has surged in popularity with the emergence of services such as ChatGPT and Google Gemini. While these models have demonstrated transformative potential in productivity and communication, they overwhelmingly cater to high-resource languages like English. This has amplified concerns over linguistic inequality in natural language processing (NLP). This paper presents the first systematic review focused specifically on strategies to address data scarcity in generative language modelling for low-resource languages (LRL). Drawing from 54 studies, we identify, categorise and evaluate technical approaches, including monolingual data augmentation, back-translation, multilingual training, and prompt engineering, across generative tasks. We also analyse trends in architecture choices, language family representation, and evaluation methods. Our findings highlight a strong reliance on transformer-based models, a concentration on a small subset of LRLs, and a lack of consistent evaluation across studies. We conclude with recommendations for extending these methods to a wider range of LRLs and outline open challenges in building equitable generative language systems. Ultimately, this review aims to support researchers and developers in building inclusive AI tools for underrepresented languages, a necessary step toward empowering LRL speakers and the preservation of linguistic diversity in a world increasingly shaped by large-scale language technologies. 

---
# Pangu Ultra MoE: How to Train Your Big MoE on Ascend NPUs 

**Authors**: Yehui Tang, Yichun Yin, Yaoyuan Wang, Hang Zhou, Yu Pan, Wei Guo, Ziyang Zhang, Miao Rang, Fangcheng Liu, Naifu Zhang, Binghan Li, Yonghan Dong, Xiaojun Meng, Yasheng Wang, Dong Li, Yin Li, Dandan Tu, Can Chen, Youliang Yan, Fisher Yu, Ruiming Tang, Yunhe Wang, Botian Huang, Bo Wang, Boxiao Liu, Changzheng Zhang, Da Kuang, Fei Liu, Gang Huang, Jiansheng Wei, Jiarui Qin, Jie Ran, Jinpeng Li, Jun Zhao, Liang Dai, Lin Li, Liqun Deng, Peifeng Qin, Pengyuan Zeng, Qiang Gu, Shaohua Tang, Shengjun Cheng, Tao Gao, Tao Yu, Tianshu Li, Tianyu Bi, Wei He, Weikai Mao, Wenyong Huang, Wulong Liu, Xiabing Li, Xianzhi Yu, Xueyu Wu, Xu He, Yangkai Du, Yan Xu, Ye Tian, Yimeng Wu, Yongbing Huang, Yong Tian, Yong Zhu, Yue Li, Yufei Wang, Yuhang Gai, Yujun Li, Yu Luo, Yunsheng Ni, Yusen Sun, Zelin Chen, Zhe Liu, Zhicheng Liu, Zhipeng Tu, Zilin Ding, Zongyuan Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2505.04519)  

**Abstract**: Sparse large language models (LLMs) with Mixture of Experts (MoE) and close to a trillion parameters are dominating the realm of most capable language models. However, the massive model scale poses significant challenges for the underlying software and hardware systems. In this paper, we aim to uncover a recipe to harness such scale on Ascend NPUs. The key goals are better usage of the computing resources under the dynamic sparse model structures and materializing the expected performance gain on the actual hardware. To select model configurations suitable for Ascend NPUs without repeatedly running the expensive experiments, we leverage simulation to compare the trade-off of various model hyperparameters. This study led to Pangu Ultra MoE, a sparse LLM with 718 billion parameters, and we conducted experiments on the model to verify the simulation results. On the system side, we dig into Expert Parallelism to optimize the communication between NPU devices to reduce the synchronization overhead. We also optimize the memory efficiency within the devices to further reduce the parameter and activation management overhead. In the end, we achieve an MFU of 30.0% when training Pangu Ultra MoE, with performance comparable to that of DeepSeek R1, on 6K Ascend NPUs, and demonstrate that the Ascend system is capable of harnessing all the training stages of the state-of-the-art language models. Extensive experiments indicate that our recipe can lead to efficient training of large-scale sparse language models with MoE. We also study the behaviors of such models for future reference. 

---
# Detecting Spelling and Grammatical Anomalies in Russian Poetry Texts 

**Authors**: Ilya Koziev  

**Link**: [PDF](https://arxiv.org/pdf/2505.04507)  

**Abstract**: The quality of natural language texts in fine-tuning datasets plays a critical role in the performance of generative models, particularly in computational creativity tasks such as poem or song lyric generation. Fluency defects in generated poems significantly reduce their value. However, training texts are often sourced from internet-based platforms without stringent quality control, posing a challenge for data engineers to manage defect levels effectively.
To address this issue, we propose the use of automated linguistic anomaly detection to identify and filter out low-quality texts from training datasets for creative models. In this paper, we present a comprehensive comparison of unsupervised and supervised text anomaly detection approaches, utilizing both synthetic and human-labeled datasets. We also introduce the RUPOR dataset, a collection of Russian-language human-labeled poems designed for cross-sentence grammatical error detection, and provide the full evaluation code. Our work aims to empower the community with tools and insights to improve the quality of training datasets for generative models in creative domains. 

---
# OBLIVIATE: Robust and Practical Machine Unlearning for Large Language Models 

**Authors**: Xiaoyu Xu, Minxin Du, Qingqing Ye, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04416)  

**Abstract**: Large language models (LLMs) trained over extensive corpora risk memorizing sensitive, copyrighted, or toxic content. To address this, we propose OBLIVIATE, a robust unlearning framework that removes targeted data while preserving model utility. The framework follows a structured process: extracting target tokens, building retain sets, and fine-tuning with a tailored loss function comprising three components -- masking, distillation, and world fact. Using low-rank adapters (LoRA), it ensures efficiency without compromising unlearning quality. We conduct experiments on multiple datasets, including the Harry Potter series, WMDP, and TOFU, using a comprehensive suite of metrics: forget quality (new document-level memorization score), model utility, and fluency. Results demonstrate its effectiveness in resisting membership inference attacks, minimizing the impact on retained data, and maintaining robustness across diverse scenarios. 

---
# YABLoCo: Yet Another Benchmark for Long Context Code Generation 

**Authors**: Aidar Valeev, Roman Garaev, Vadim Lomshakov, Irina Piontkovskaya, Vladimir Ivanov, Israel Adewuyi  

**Link**: [PDF](https://arxiv.org/pdf/2505.04406)  

**Abstract**: Large Language Models demonstrate the ability to solve various programming tasks, including code generation. Typically, the performance of LLMs is measured on benchmarks with small or medium-sized context windows of thousands of lines of code. At the same time, in real-world software projects, repositories can span up to millions of LoC. This paper closes this gap by contributing to the long context code generation benchmark (YABLoCo). The benchmark featured a test set of 215 functions selected from four large repositories with thousands of functions. The dataset contained metadata of functions, contexts of the functions with different levels of dependencies, docstrings, functions bodies, and call graphs for each repository. This paper presents three key aspects of the contribution. First, the benchmark aims at function body generation in large repositories in C and C++, two languages not covered by previous benchmarks. Second, the benchmark contains large repositories from 200K to 2,000K LoC. Third, we contribute a scalable evaluation pipeline for efficient computing of the target metrics and a tool for visual analysis of generated code. Overall, these three aspects allow for evaluating code generation in large repositories in C and C++. 

---
# Large Means Left: Political Bias in Large Language Models Increases with Their Number of Parameters 

**Authors**: David Exler, Mark Schutera, Markus Reischl, Luca Rettenberger  

**Link**: [PDF](https://arxiv.org/pdf/2505.04393)  

**Abstract**: With the increasing prevalence of artificial intelligence, careful evaluation of inherent biases needs to be conducted to form the basis for alleviating the effects these predispositions can have on users. Large language models (LLMs) are predominantly used by many as a primary source of information for various topics. LLMs frequently make factual errors, fabricate data (hallucinations), or present biases, exposing users to misinformation and influencing opinions. Educating users on their risks is key to responsible use, as bias, unlike hallucinations, cannot be caught through data verification. We quantify the political bias of popular LLMs in the context of the recent vote of the German Bundestag using the score produced by the Wahl-O-Mat. This metric measures the alignment between an individual's political views and the positions of German political parties. We compare the models' alignment scores to identify factors influencing their political preferences. Doing so, we discover a bias toward left-leaning parties, most dominant in larger LLMs. Also, we find that the language we use to communicate with the models affects their political views. Additionally, we analyze the influence of a model's origin and release date and compare the results to the outcome of the recent vote of the Bundestag. Our results imply that LLMs are prone to exhibiting political bias. Large corporations with the necessary means to develop LLMs, thus, knowingly or unknowingly, have a responsibility to contain these biases, as they can influence each voter's decision-making process and inform public opinion in general and at scale. 

---
# The Aloe Family Recipe for Open and Specialized Healthcare LLMs 

**Authors**: Dario Garcia-Gasulla, Jordi Bayarri-Planas, Ashwin Kumar Gururajan, Enrique Lopez-Cuena, Adrian Tormos, Daniel Hinjos, Pablo Bernabeu-Perez, Anna Arias-Duart, Pablo Agustin Martin-Torres, Marta Gonzalez-Mallo, Sergio Alvarez-Napagao, Eduard Ayguadé-Parra, Ulises Cortés  

**Link**: [PDF](https://arxiv.org/pdf/2505.04388)  

**Abstract**: Purpose: With advancements in Large Language Models (LLMs) for healthcare, the need arises for competitive open-source models to protect the public interest. This work contributes to the field of open medical LLMs by optimizing key stages of data preprocessing and training, while showing how to improve model safety (through DPO) and efficacy (through RAG). The evaluation methodology used, which includes four different types of tests, defines a new standard for the field. The resultant models, shown to be competitive with the best private alternatives, are released with a permisive license.
Methods: Building on top of strong base models like Llama 3.1 and Qwen 2.5, Aloe Beta uses a custom dataset to enhance public data with synthetic Chain of Thought examples. The models undergo alignment with Direct Preference Optimization, emphasizing ethical and policy-aligned performance in the presence of jailbreaking attacks. Evaluation includes close-ended, open-ended, safety and human assessments, to maximize the reliability of results.
Results: Recommendations are made across the entire pipeline, backed by the solid performance of the Aloe Family. These models deliver competitive performance across healthcare benchmarks and medical fields, and are often preferred by healthcare professionals. On bias and toxicity, the Aloe Beta models significantly improve safety, showing resilience to unseen jailbreaking attacks. For a responsible release, a detailed risk assessment specific to healthcare is attached to the Aloe Family models.
Conclusion: The Aloe Beta models, and the recipe that leads to them, are a significant contribution to the open-source medical LLM field, offering top-of-the-line performance while maintaining high ethical requirements. This work sets a new standard for developing and reporting aligned LLMs in healthcare. 

---
# GASCADE: Grouped Summarization of Adverse Drug Event for Enhanced Cancer Pharmacovigilance 

**Authors**: Sofia Jamil, Aryan Dabad, Bollampalli Areen Reddy, Sriparna Saha, Rajiv Misra, Adil A. Shakur  

**Link**: [PDF](https://arxiv.org/pdf/2505.04284)  

**Abstract**: In the realm of cancer treatment, summarizing adverse drug events (ADEs) reported by patients using prescribed drugs is crucial for enhancing pharmacovigilance practices and improving drug-related decision-making. While the volume and complexity of pharmacovigilance data have increased, existing research in this field has predominantly focused on general diseases rather than specifically addressing cancer. This work introduces the task of grouped summarization of adverse drug events reported by multiple patients using the same drug for cancer treatment. To address the challenge of limited resources in cancer pharmacovigilance, we present the MultiLabeled Cancer Adverse Drug Reaction and Summarization (MCADRS) dataset. This dataset includes pharmacovigilance posts detailing patient concerns regarding drug efficacy and adverse effects, along with extracted labels for drug names, adverse drug events, severity, and adversity of reactions, as well as summaries of ADEs for each drug. Additionally, we propose the Grouping and Abstractive Summarization of Cancer Adverse Drug events (GASCADE) framework, a novel pipeline that combines the information extraction capabilities of Large Language Models (LLMs) with the summarization power of the encoder-decoder T5 model. Our work is the first to apply alignment techniques, including advanced algorithms like Direct Preference Optimization, to encoder-decoder models using synthetic datasets for summarization tasks. Through extensive experiments, we demonstrate the superior performance of GASCADE across various metrics, validated through both automated assessments and human evaluations. This multitasking approach enhances drug-related decision-making and fosters a deeper understanding of patient concerns, paving the way for advancements in personalized and responsive cancer care. The code and dataset used in this work are publicly available. 

---
# LLM-Independent Adaptive RAG: Let the Question Speak for Itself 

**Authors**: Maria Marina, Nikolay Ivanov, Sergey Pletenev, Mikhail Salnikov, Daria Galimzianova, Nikita Krayko, Vasily Konovalov, Alexander Panchenko, Viktor Moskvoretskii  

**Link**: [PDF](https://arxiv.org/pdf/2505.04253)  

**Abstract**: Large Language Models~(LLMs) are prone to hallucinations, and Retrieval-Augmented Generation (RAG) helps mitigate this, but at a high computational cost while risking misinformation. Adaptive retrieval aims to retrieve only when necessary, but existing approaches rely on LLM-based uncertainty estimation, which remain inefficient and impractical. In this study, we introduce lightweight LLM-independent adaptive retrieval methods based on external information. We investigated 27 features, organized into 7 groups, and their hybrid combinations. We evaluated these methods on 6 QA datasets, assessing the QA performance and efficiency. The results show that our approach matches the performance of complex LLM-based methods while achieving significant efficiency gains, demonstrating the potential of external information for adaptive retrieval. 

---
# Can Language Models Understand Social Behavior in Clinical Conversations? 

**Authors**: Manas Satish Bedmutha, Feng Chen, Andrea Hartzler, Trevor Cohen, Nadir Weibel  

**Link**: [PDF](https://arxiv.org/pdf/2505.04152)  

**Abstract**: Effective communication between providers and their patients influences health and care outcomes. The effectiveness of such conversations has been linked not only to the exchange of clinical information, but also to a range of interpersonal behaviors; commonly referred to as social signals, which are often conveyed through non-verbal cues and shape the quality of the patient-provider relationship. Recent advances in large language models (LLMs) have demonstrated an increasing ability to infer emotional and social behaviors even when analyzing only textual information. As automation increases also in clinical settings, such as for transcription of patient-provider conversations, there is growing potential for LLMs to automatically analyze and extract social behaviors from these interactions. To explore the foundational capabilities of LLMs in tracking social signals in clinical dialogue, we designed task-specific prompts and evaluated model performance across multiple architectures and prompting styles using a highly imbalanced, annotated dataset spanning 20 distinct social signals such as provider dominance, patient warmth, etc. We present the first system capable of tracking all these 20 coded signals, and uncover patterns in LLM behavior. Further analysis of model configurations and clinical context provides insights for enhancing LLM performance on social signal processing tasks in healthcare settings. 

---
# Unmasking the Canvas: A Dynamic Benchmark for Image Generation Jailbreaking and LLM Content Safety 

**Authors**: Variath Madhupal Gautham Nair, Vishal Varma Dantuluri  

**Link**: [PDF](https://arxiv.org/pdf/2505.04146)  

**Abstract**: Existing large language models (LLMs) are advancing rapidly and produce outstanding results in image generation tasks, yet their content safety checks remain vulnerable to prompt-based jailbreaks. Through preliminary testing on platforms such as ChatGPT, MetaAI, and Grok, we observed that even short, natural prompts could lead to the generation of compromising images ranging from realistic depictions of forged documents to manipulated images of public figures.
We introduce Unmasking the Canvas (UTC Benchmark; UTCB), a dynamic and scalable benchmark dataset to evaluate LLM vulnerability in image generation. Our methodology combines structured prompt engineering, multilingual obfuscation (e.g., Zulu, Gaelic, Base64), and evaluation using Groq-hosted LLaMA-3. The pipeline supports both zero-shot and fallback prompting strategies, risk scoring, and automated tagging. All generations are stored with rich metadata and curated into Bronze (non-verified), Silver (LLM-aided verification), and Gold (manually verified) tiers. UTCB is designed to evolve over time with new data sources, prompt templates, and model behaviors.
Warning: This paper includes visual examples of adversarial inputs designed to test model safety. All outputs have been redacted to ensure responsible disclosure. 

---
# Enhancing Granular Sentiment Classification with Chain-of-Thought Prompting in Large Language Models 

**Authors**: Vihaan Miriyala, Smrithi Bukkapatnam, Lavanya Prahallad  

**Link**: [PDF](https://arxiv.org/pdf/2505.04135)  

**Abstract**: We explore the use of Chain-of-Thought (CoT) prompting with large language models (LLMs) to improve the accuracy of granular sentiment categorization in app store reviews. Traditional numeric and polarity-based ratings often fail to capture the nuanced sentiment embedded in user feedback. We evaluated the effectiveness of CoT prompting versus simple prompting on 2000 Amazon app reviews by comparing each method's predictions to human judgements. CoT prompting improved classification accuracy from 84% to 93% highlighting the benefit of explicit reasoning in enhancing sentiment analysis performance. 

---
# Bringing legal knowledge to the public by constructing a legal question bank using large-scale pre-trained language model 

**Authors**: Mingruo Yuan, Ben Kao, Tien-Hsuan Wu, Michael M. K. Cheung, Henry W. H. Chan, Anne S. Y. Cheung, Felix W. H. Chan, Yongxi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.04132)  

**Abstract**: Access to legal information is fundamental to access to justice. Yet accessibility refers not only to making legal documents available to the public, but also rendering legal information comprehensible to them. A vexing problem in bringing legal information to the public is how to turn formal legal documents such as legislation and judgments, which are often highly technical, to easily navigable and comprehensible knowledge to those without legal education. In this study, we formulate a three-step approach for bringing legal knowledge to laypersons, tackling the issues of navigability and comprehensibility. First, we translate selected sections of the law into snippets (called CLIC-pages), each being a small piece of article that focuses on explaining certain technical legal concept in layperson's terms. Second, we construct a Legal Question Bank (LQB), which is a collection of legal questions whose answers can be found in the CLIC-pages. Third, we design an interactive CLIC Recommender (CRec). Given a user's verbal description of a legal situation that requires a legal solution, CRec interprets the user's input and shortlists questions from the question bank that are most likely relevant to the given legal situation and recommends their corresponding CLIC pages where relevant legal knowledge can be found. In this paper we focus on the technical aspects of creating an LQB. We show how large-scale pre-trained language models, such as GPT-3, can be used to generate legal questions. We compare machine-generated questions (MGQs) against human-composed questions (HCQs) and find that MGQs are more scalable, cost-effective, and more diversified, while HCQs are more precise. We also show a prototype of CRec and illustrate through an example how our 3-step approach effectively brings relevant legal knowledge to the public. 

---
# Natural Language Generation in Healthcare: A Review of Methods and Applications 

**Authors**: Mengxian Lyu, Xiaohan Li, Ziyi Chen, Jinqian Pan, Cheng Peng, Sankalp Talankar, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04073)  

**Abstract**: Natural language generation (NLG) is the key technology to achieve generative artificial intelligence (AI). With the breakthroughs in large language models (LLMs), NLG has been widely used in various medical applications, demonstrating the potential to enhance clinical workflows, support clinical decision-making, and improve clinical documentation. Heterogeneous and diverse medical data modalities, such as medical text, images, and knowledge bases, are utilized in NLG. Researchers have proposed many generative models and applied them in a number of healthcare applications. There is a need for a comprehensive review of NLG methods and applications in the medical domain. In this study, we systematically reviewed 113 scientific publications from a total of 3,988 NLG-related articles identified using a literature search, focusing on data modality, model architecture, clinical applications, and evaluation methods. Following PRISMA (Preferred Reporting Items for Systematic reviews and Meta-Analyses) guidelines, we categorize key methods, identify clinical applications, and assess their capabilities, limitations, and emerging challenges. This timely review covers the key NLG technologies and medical applications and provides valuable insights for future studies to leverage NLG to transform medical discovery and healthcare. 

---
# Advancing and Benchmarking Personalized Tool Invocation for LLMs 

**Authors**: Xu Huang, Yuefeng Huang, Weiwen Liu, Xingshan Zeng, Yasheng Wang, Ruiming Tang, Hong Xie, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2505.04072)  

**Abstract**: Tool invocation is a crucial mechanism for extending the capabilities of Large Language Models (LLMs) and has recently garnered significant attention. It enables LLMs to solve complex problems through tool calls while accessing up-to-date world knowledge. However, existing work primarily focuses on the fundamental ability of LLMs to invoke tools for problem-solving, without considering personalized constraints in tool invocation. In this work, we introduce the concept of Personalized Tool Invocation and define two key tasks: Tool Preference and Profile-dependent Query. Tool Preference addresses user preferences when selecting among functionally similar tools, while Profile-dependent Query considers cases where a user query lacks certain tool parameters, requiring the model to infer them from the user profile. To tackle these challenges, we propose PTool, a data synthesis framework designed for personalized tool invocation. Additionally, we construct \textbf{PTBench}, the first benchmark for evaluating personalized tool invocation. We then fine-tune various open-source models, demonstrating the effectiveness of our framework and providing valuable insights. Our benchmark is public at this https URL. 

---
# SLOT: Structuring the Output of Large Language Models 

**Authors**: Darren Yow-Bang Wang, Zhengyuan Shen, Soumya Smruti Mishra, Zhichao Xu, Yifei Teng, Haibo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.04016)  

**Abstract**: Structured outputs are essential for large language models (LLMs) in critical applications like agents and information extraction. Despite their capabilities, LLMs often generate outputs that deviate from predefined schemas, significantly hampering reliable application development. We present SLOT (Structured LLM Output Transformer), a model-agnostic approach that transforms unstructured LLM outputs into precise structured formats. While existing solutions predominantly rely on constrained decoding techniques or are tightly coupled with specific models, SLOT employs a fine-tuned lightweight language model as a post-processing layer, achieving flexibility across various LLMs and schema specifications. We introduce a systematic pipeline for data curation and synthesis alongside a formal evaluation methodology that quantifies both schema accuracy and content fidelity. Our results demonstrate that fine-tuned Mistral-7B model with constrained decoding achieves near perfect schema accuracy (99.5%) and content similarity (94.0%), outperforming Claude-3.5-Sonnet by substantial margins (+25 and +20 percentage points, respectively). Notably, even compact models like Llama-3.2-1B can match or exceed the structured output capabilities of much larger proprietary models when equipped with SLOT, enabling reliable structured generation in resource-constrained environments. 

---
# X-Reasoner: Towards Generalizable Reasoning Across Modalities and Domains 

**Authors**: Qianchu Liu, Sheng Zhang, Guanghui Qin, Timothy Ossowski, Yu Gu, Ying Jin, Sid Kiblawi, Sam Preston, Mu Wei, Paul Vozila, Tristan Naumann, Hoifung Poon  

**Link**: [PDF](https://arxiv.org/pdf/2505.03981)  

**Abstract**: Recent proprietary models (e.g., o3) have begun to demonstrate strong multimodal reasoning capabilities. Yet, most existing open-source research concentrates on training text-only reasoning models, with evaluations limited to mainly mathematical and general-domain tasks. Therefore, it remains unclear how to effectively extend reasoning capabilities beyond text input and general domains. This paper explores a fundamental research question: Is reasoning generalizable across modalities and domains? Our findings support an affirmative answer: General-domain text-based post-training can enable such strong generalizable reasoning. Leveraging this finding, we introduce X-Reasoner, a vision-language model post-trained solely on general-domain text for generalizable reasoning, using a two-stage approach: an initial supervised fine-tuning phase with distilled long chain-of-thoughts, followed by reinforcement learning with verifiable rewards. Experiments show that X-Reasoner successfully transfers reasoning capabilities to both multimodal and out-of-domain settings, outperforming existing state-of-the-art models trained with in-domain and multimodal data across various general and medical benchmarks (Figure 1). Additionally, we find that X-Reasoner's performance in specialized domains can be further enhanced through continued training on domain-specific text-only data. Building upon this, we introduce X-Reasoner-Med, a medical-specialized variant that achieves new state of the art on numerous text-only and multimodal medical benchmarks. 

---
# Divide, Optimize, Merge: Fine-Grained LLM Agent Optimization at Scale 

**Authors**: Jiale Liu, Yifan Zeng, Shaokun Zhang, Chi Zhang, Malte Højmark-Bertelsen, Marie Normann Gadeberg, Huazheng Wang, Qingyun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03973)  

**Abstract**: LLM-based optimization has shown remarkable potential in enhancing agentic systems. However, the conventional approach of prompting LLM optimizer with the whole training trajectories on training dataset in a single pass becomes untenable as datasets grow, leading to context window overflow and degraded pattern recognition. To address these challenges, we propose Fine-Grained Optimization (FGO), a scalable framework that divides large optimization tasks into manageable subsets, performs targeted optimizations, and systematically combines optimized components through progressive merging. Evaluation across ALFWorld, LogisticsQA, and GAIA benchmarks demonstrate that FGO outperforms existing approaches by 1.6-8.6% while reducing average prompt token consumption by 56.3%. Our framework provides a practical solution for scaling up LLM-based optimization of increasingly sophisticated agent systems. Further analysis demonstrates that FGO achieves the most consistent performance gain in all training dataset sizes, showcasing its scalability and efficiency. 

---
# A Reasoning-Focused Legal Retrieval Benchmark 

**Authors**: Lucia Zheng, Neel Guha, Javokhir Arifov, Sarah Zhang, Michal Skreta, Christopher D. Manning, Peter Henderson, Daniel E. Ho  

**Link**: [PDF](https://arxiv.org/pdf/2505.03970)  

**Abstract**: As the legal community increasingly examines the use of large language models (LLMs) for various legal applications, legal AI developers have turned to retrieval-augmented LLMs ("RAG" systems) to improve system performance and robustness. An obstacle to the development of specialized RAG systems is the lack of realistic legal RAG benchmarks which capture the complexity of both legal retrieval and downstream legal question-answering. To address this, we introduce two novel legal RAG benchmarks: Bar Exam QA and Housing Statute QA. Our tasks correspond to real-world legal research tasks, and were produced through annotation processes which resemble legal research. We describe the construction of these benchmarks and the performance of existing retriever pipelines. Our results suggest that legal RAG remains a challenging application, thus motivating future research. 

---
# Hesitation is defeat? Connecting Linguistic and Predictive Uncertainty 

**Authors**: Gianluca Manzo, Julia Ive  

**Link**: [PDF](https://arxiv.org/pdf/2505.03910)  

**Abstract**: Automating chest radiograph interpretation using Deep Learning (DL) models has the potential to significantly improve clinical workflows, decision-making, and large-scale health screening. However, in medical settings, merely optimising predictive performance is insufficient, as the quantification of uncertainty is equally crucial. This paper investigates the relationship between predictive uncertainty, derived from Bayesian Deep Learning approximations, and human/linguistic uncertainty, as estimated from free-text radiology reports labelled by rule-based labellers. Utilising BERT as the model of choice, this study evaluates different binarisation methods for uncertainty labels and explores the efficacy of Monte Carlo Dropout and Deep Ensembles in estimating predictive uncertainty. The results demonstrate good model performance, but also a modest correlation between predictive and linguistic uncertainty, highlighting the challenges in aligning machine uncertainty with human interpretation nuances. Our findings suggest that while Bayesian approximations provide valuable uncertainty estimates, further refinement is necessary to fully capture and utilise the subtleties of human uncertainty in clinical applications. 

---
# Calibrating Uncertainty Quantification of Multi-Modal LLMs using Grounding 

**Authors**: Trilok Padhi, Ramneet Kaur, Adam D. Cobb, Manoj Acharya, Anirban Roy, Colin Samplawski, Brian Matejek, Alexander M. Berenbeim, Nathaniel D. Bastian, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2505.03788)  

**Abstract**: We introduce a novel approach for calibrating uncertainty quantification (UQ) tailored for multi-modal large language models (LLMs). Existing state-of-the-art UQ methods rely on consistency among multiple responses generated by the LLM on an input query under diverse settings. However, these approaches often report higher confidence in scenarios where the LLM is consistently incorrect. This leads to a poorly calibrated confidence with respect to accuracy. To address this, we leverage cross-modal consistency in addition to self-consistency to improve the calibration of the multi-modal models. Specifically, we ground the textual responses to the visual inputs. The confidence from the grounding model is used to calibrate the overall confidence. Given that using a grounding model adds its own uncertainty in the pipeline, we apply temperature scaling - a widely accepted parametric calibration technique - to calibrate the grounding model's confidence in the accuracy of generated responses. We evaluate the proposed approach across multiple multi-modal tasks, such as medical question answering (Slake) and visual question answering (VQAv2), considering multi-modal models such as LLaVA-Med and LLaVA. The experiments demonstrate that the proposed framework achieves significantly improved calibration on both tasks. 

---
# Beyond Theorem Proving: Formulation, Framework and Benchmark for Formal Problem-Solving 

**Authors**: Qi Liu, Xinhao Zheng, Renqiu Xia, Xingzhi Qi, Qinxiang Cao, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.04528)  

**Abstract**: As a seemingly self-explanatory task, problem-solving has been a significant component of science and engineering. However, a general yet concrete formulation of problem-solving itself is missing. With the recent development of AI-based problem-solving agents, the demand for process-level verifiability is rapidly increasing yet underexplored. To fill these gaps, we present a principled formulation of problem-solving as a deterministic Markov decision process; a novel framework, FPS (Formal Problem-Solving), which utilizes existing FTP (formal theorem proving) environments to perform process-verified problem-solving; and D-FPS (Deductive FPS), decoupling solving and answer verification for better human-alignment. The expressiveness, soundness and completeness of the frameworks are proven. We construct three benchmarks on problem-solving: FormalMath500, a formalization of a subset of the MATH500 benchmark; MiniF2F-Solving and PutnamBench-Solving, adaptations of FTP benchmarks MiniF2F and PutnamBench. For faithful, interpretable, and human-aligned evaluation, we propose RPE (Restricted Propositional Equivalence), a symbolic approach to determine the correctness of answers by formal verification. We evaluate four prevalent FTP models and two prompting methods as baselines, solving at most 23.77% of FormalMath500, 27.47% of MiniF2F-Solving, and 0.31% of PutnamBench-Solving. 

---
# Miipher-2: A Universal Speech Restoration Model for Million-Hour Scale Data Restoration 

**Authors**: Shigeki Karita, Yuma Koizumi, Heiga Zen, Haruko Ishikawa, Robin Scheibler, Michiel Bacchiani  

**Link**: [PDF](https://arxiv.org/pdf/2505.04457)  

**Abstract**: Training data cleaning is a new application for generative model-based speech restoration (SR). This paper introduces Miipher-2, an SR model designed for million-hour scale data, for training data cleaning for large-scale generative models like large language models. Key challenges addressed include generalization to unseen languages, operation without explicit conditioning (e.g., text, speaker ID), and computational efficiency. Miipher-2 utilizes a frozen, pre-trained Universal Speech Model (USM), supporting over 300 languages, as a robust, conditioning-free feature extractor. To optimize efficiency and minimize memory, Miipher-2 incorporates parallel adapters for predicting clean USM features from noisy inputs and employs the WaneFit neural vocoder for waveform synthesis. These components were trained on 3,000 hours of multi-lingual, studio-quality recordings with augmented degradations, while USM parameters remained fixed. Experimental results demonstrate Miipher-2's superior or comparable performance to conventional SR models in word-error-rate, speaker similarity, and both objective and subjective sound quality scores across all tested languages. Miipher-2 operates efficiently on consumer-grade accelerators, achieving a real-time factor of 0.0078, enabling the processing of a million-hour speech dataset in approximately three days using only 100 such accelerators. 

---
# Benchmarking LLMs' Swarm intelligence 

**Authors**: Kai Ruan, Mowen Huang, Ji-Rong Wen, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.04364)  

**Abstract**: Large Language Models (LLMs) show potential for complex reasoning, yet their capacity for emergent coordination in Multi-Agent Systems (MAS) when operating under strict constraints-such as limited local perception and communication, characteristic of natural swarms-remains largely unexplored, particularly concerning the nuances of swarm intelligence. Existing benchmarks often do not fully capture the unique challenges of decentralized coordination that arise when agents operate with incomplete spatio-temporal information. To bridge this gap, we introduce SwarmBench, a novel benchmark designed to systematically evaluate the swarm intelligence capabilities of LLMs acting as decentralized agents. SwarmBench features five foundational MAS coordination tasks within a configurable 2D grid environment, forcing agents to rely primarily on local sensory input (k x k view) and local communication. We propose metrics for coordination effectiveness and analyze emergent group dynamics. Evaluating several leading LLMs in a zero-shot setting, we find significant performance variations across tasks, highlighting the difficulties posed by local information constraints. While some coordination emerges, results indicate limitations in robust planning and strategy formation under uncertainty in these decentralized scenarios. Assessing LLMs under swarm-like conditions is crucial for realizing their potential in future decentralized systems. We release SwarmBench as an open, extensible toolkit-built upon a customizable and scalable physical system with defined mechanical properties. It provides environments, prompts, evaluation scripts, and the comprehensive experimental datasets generated, aiming to foster reproducible research into LLM-based MAS coordination and the theoretical underpinnings of Embodied MAS. Our code repository is available at this https URL. 

---
# VideoPath-LLaVA: Pathology Diagnostic Reasoning Through Video Instruction Tuning 

**Authors**: Trinh T.L. Vuong, Jin Tae Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2505.04192)  

**Abstract**: We present VideoPath-LLaVA, the first large multimodal model (LMM) in computational pathology that integrates three distinct image scenarios, single patch images, automatically keyframe-extracted clips, and manually segmented video pathology images, to mimic the natural diagnostic process of pathologists. By generating detailed histological descriptions and culminating in a definitive sign-out diagnosis, VideoPath-LLaVA bridges visual narratives with diagnostic reasoning.
Central to our approach is the VideoPath-Instruct dataset, comprising 4278 video and diagnosis-specific chain-of-thought instructional pairs sourced from educational histopathology videos on YouTube. Although high-quality data is critical for enhancing diagnostic reasoning, its creation is time-intensive and limited in volume. To overcome this challenge, we transfer knowledge from existing single-image instruction datasets to train on weakly annotated, keyframe-extracted clips, followed by fine-tuning on manually segmented videos. VideoPath-LLaVA establishes a new benchmark in pathology video analysis and offers a promising foundation for future AI systems that support clinical decision-making through integrated visual and diagnostic reasoning. Our code, data, and model are publicly available at this https URL. 

---
# Large Language Models are often politically extreme, usually ideologically inconsistent, and persuasive even in informational contexts 

**Authors**: Nouar Aldahoul, Hazem Ibrahim, Matteo Varvello, Aaron Kaufman, Talal Rahwan, Yasir Zaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.04171)  

**Abstract**: Large Language Models (LLMs) are a transformational technology, fundamentally changing how people obtain information and interact with the world. As people become increasingly reliant on them for an enormous variety of tasks, a body of academic research has developed to examine these models for inherent biases, especially political biases, often finding them small. We challenge this prevailing wisdom. First, by comparing 31 LLMs to legislators, judges, and a nationally representative sample of U.S. voters, we show that LLMs' apparently small overall partisan preference is the net result of offsetting extreme views on specific topics, much like moderate voters. Second, in a randomized experiment, we show that LLMs can promulgate their preferences into political persuasiveness even in information-seeking contexts: voters randomized to discuss political issues with an LLM chatbot are as much as 5 percentage points more likely to express the same preferences as that chatbot. Contrary to expectations, these persuasive effects are not moderated by familiarity with LLMs, news consumption, or interest in politics. LLMs, especially those controlled by private companies or governments, may become a powerful and targeted vector for political influence. 

---
# LLAMAPIE: Proactive In-Ear Conversation Assistants 

**Authors**: Tuochao Chen, Nicholas Batchelder, Alisa Liu, Noah Smith, Shyamnath Gollakota  

**Link**: [PDF](https://arxiv.org/pdf/2505.04066)  

**Abstract**: We introduce LlamaPIE, the first real-time proactive assistant designed to enhance human conversations through discreet, concise guidance delivered via hearable devices. Unlike traditional language models that require explicit user invocation, this assistant operates in the background, anticipating user needs without interrupting conversations. We address several challenges, including determining when to respond, crafting concise responses that enhance conversations, leveraging knowledge of the user for context-aware assistance, and real-time, on-device processing. To achieve this, we construct a semi-synthetic dialogue dataset and propose a two-model pipeline: a small model that decides when to respond and a larger model that generates the response. We evaluate our approach on real-world datasets, demonstrating its effectiveness in providing helpful, unobtrusive assistance. User studies with our assistant, implemented on Apple Silicon M2 hardware, show a strong preference for the proactive assistant over both a baseline with no assistance and a reactive model, highlighting the potential of LlamaPie to enhance live conversations. 

---
# Quiet Feature Learning in Algorithmic Tasks 

**Authors**: Prudhviraj Naidu, Zixian Wang, Leon Bergen, Ramamohan Paturi  

**Link**: [PDF](https://arxiv.org/pdf/2505.03997)  

**Abstract**: We train Transformer-based language models on ten foundational algorithmic tasks and observe pronounced phase transitions in their loss curves that deviate from established power-law scaling trends. Over large ranges of compute, the validation loss barely improves, then abruptly decreases. Probing the models' internal representations reveals the learning of quiet features during the stagnant phase, followed by sudden acquisition of loud features that coincide with the sharp drop in loss. Our ablation experiments show that disrupting a single learned feature can dramatically degrade performance, providing evidence of their causal role in task performance. These findings challenge the prevailing assumption that next-token predictive loss reliably tracks incremental progress; instead, key internal features may be developing below the surface until they coalesce, triggering a rapid performance gain. 

---
# The Power of Stories: Narrative Priming Shapes How LLM Agents Collaborate and Compete 

**Authors**: Gerrit Großmann, Larisa Ivanova, Sai Leela Poduru, Mohaddeseh Tabrizian, Islam Mesabah, David A. Selby, Sebastian J. Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2505.03961)  

**Abstract**: According to Yuval Noah Harari, large-scale human cooperation is driven by shared narratives that encode common beliefs and values. This study explores whether such narratives can similarly nudge LLM agents toward collaboration. We use a finitely repeated public goods game in which LLM agents choose either cooperative or egoistic spending strategies. We prime agents with stories highlighting teamwork to different degrees and test how this influences negotiation outcomes. Our experiments explore four questions:(1) How do narratives influence negotiation behavior? (2) What differs when agents share the same story versus different ones? (3) What happens when the agent numbers grow? (4) Are agents resilient against self-serving negotiators? We find that story-based priming significantly affects negotiation strategies and success rates. Common stories improve collaboration, benefiting each agent. By contrast, priming agents with different stories reverses this effect, and those agents primed toward self-interest prevail. We hypothesize that these results carry implications for multi-agent system design and AI alignment. 

---
# Sentiment-Aware Recommendation Systems in E-Commerce: A Review from a Natural Language Processing Perspective 

**Authors**: Yogesh Gajula  

**Link**: [PDF](https://arxiv.org/pdf/2505.03828)  

**Abstract**: E-commerce platforms generate vast volumes of user feedback, such as star ratings, written reviews, and comments. However, most recommendation engines rely primarily on numerical scores, often overlooking the nuanced opinions embedded in free text. This paper comprehensively reviews sentiment-aware recommendation systems from a natural language processing perspective, covering advancements from 2023 to early 2025. It highlights the benefits of integrating sentiment analysis into e-commerce recommenders to enhance prediction accuracy and explainability through detailed opinion extraction. Our survey categorizes recent work into four main approaches: deep learning classifiers that combine sentiment embeddings with user item interactions, transformer based methods for nuanced feature extraction, graph neural networks that propagate sentiment signals, and conversational recommenders that adapt in real time to user feedback. We summarize model architectures and demonstrate how sentiment flows through recommendation pipelines, impacting dialogue-based suggestions. Key challenges include handling noisy or sarcastic text, dynamic user preferences, and bias mitigation. Finally, we outline research gaps and provide a roadmap for developing smarter, fairer, and more user-centric recommendation tools. 

---
# Cer-Eval: Certifiable and Cost-Efficient Evaluation Framework for LLMs 

**Authors**: Ganghua Wang, Zhaorun Chen, Bo Li, Haifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03814)  

**Abstract**: As foundation models continue to scale, the size of trained models grows exponentially, presenting significant challenges for their evaluation. Current evaluation practices involve curating increasingly large datasets to assess the performance of large language models (LLMs). However, there is a lack of systematic analysis and guidance on determining the sufficiency of test data or selecting informative samples for evaluation. This paper introduces a certifiable and cost-efficient evaluation framework for LLMs. Our framework adapts to different evaluation objectives and outputs confidence intervals that contain true values with high probability. We use ``test sample complexity'' to quantify the number of test points needed for a certifiable evaluation and derive tight bounds on test sample complexity. Based on the developed theory, we develop a partition-based algorithm, named Cer-Eval, that adaptively selects test points to minimize the cost of LLM evaluation. Real-world experiments demonstrate that Cer-Eval can save 20% to 40% test points across various benchmarks, while maintaining an estimation error level comparable to the current evaluation process and providing a 95% confidence guarantee. 

---
# Grouped Sequency-arranged Rotation: Optimizing Rotation Transformation for Quantization for Free 

**Authors**: Euntae Choi, Sumin Song, Woosang Lim, Sungjoo Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2505.03810)  

**Abstract**: Large Language Models (LLMs) face deployment challenges due to high computational costs, and while Post-Training Quantization (PTQ) offers a solution, existing rotation-based methods struggle at very low bit-widths like 2-bit. We introduce a novel, training-free approach to construct an improved rotation matrix, addressing the limitations of current methods. The key contributions include leveraging the Walsh-Hadamard transform with sequency ordering, which clusters similar frequency components to reduce quantization error compared to standard Hadamard matrices, significantly improving performance. Furthermore, we propose a Grouped Sequency-arranged Rotation (GSR) using block-diagonal matrices with smaller Walsh blocks, effectively isolating outlier impacts and achieving performance comparable to optimization-based methods without requiring any training. Our method demonstrates robust performance on reasoning tasks and Perplexity (PPL) score on WikiText-2. Our method also enhances results even when applied over existing learned rotation techniques. 

---
# Scalability Matters: Overcoming Challenges in InstructGLM with Similarity-Degree-Based Sampling 

**Authors**: Hyun Lee, Chris Yi, Maminur Islam, B.D.S. Aritra  

**Link**: [PDF](https://arxiv.org/pdf/2505.03799)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in various natural language processing tasks; however, their application to graph-related problems remains limited, primarily due to scalability constraints and the absence of dedicated mechanisms for processing graph structures. Existing approaches predominantly integrate LLMs with Graph Neural Networks (GNNs), using GNNs as feature encoders or auxiliary components. However, directly encoding graph structures within LLMs has been underexplored, particularly in the context of large-scale graphs where token limitations hinder effective representation. To address these challenges, we propose SDM-InstructGLM, a novel instruction-tuned Graph Language Model (InstructGLM) framework that enhances scalability and efficiency without relying on GNNs. Our method introduces a similarity-degree-based biased random walk mechanism, which selectively samples and encodes graph information based on node-feature similarity and degree centrality, ensuring an adaptive and structured representation within the LLM. This approach significantly improves token efficiency, mitigates information loss due to random sampling, and enhances performance on graph-based tasks such as node classification and link prediction. Furthermore, our results demonstrate the feasibility of LLM-only graph processing, enabling scalable and interpretable Graph Language Models (GLMs) optimized through instruction-based fine-tuning. This work paves the way for GNN-free approaches to graph learning, leveraging LLMs as standalone graph reasoning models. Our source code is available on GitHub. 

---
# When Reasoning Beats Scale: A 1.5B Reasoning Model Outranks 13B LLMs as Discriminator 

**Authors**: Md Fahim Anjum  

**Link**: [PDF](https://arxiv.org/pdf/2505.03786)  

**Abstract**: Large Language Models (LLM) with reasoning capabilities offer a promising path for improving candidate evaluation in planning frameworks, but their relative performance against traditional non-reasoning models remains largely underexplored. In this study, we benchmark a distilled 1.5B parameter reasoning model (DeepSeek-R1) against several state-of-the-art non-reasoning LLMs within a generator-discriminator LLM planning framework for the text-to-SQL task. For this, we introduce a novel method for extracting soft scores from the chain-of-thought (CoT) outputs from reasoning that enables fine-grained ranking of candidates. Our central hypothesis is that reasoning models are more effective discriminators than non-reasoning LLMs. Our results show that distilled DeepSeek-R1-1.5B achieves up to $87\%$ higher F1 and $3.7\%$ better discrimination accuracy than CodeLlama-7B, as well as $3.7\%$ higher execution accuracy than CodeLlama-13B, despite having significantly fewer parameters. Furthermore, we find that there is a limit to the logical capabilities of reasoning models, and only providing more context or allowing more compute budget for reasoning is not enough to improve their discrimination performance. Finally, we demonstrate that, unlike non-reasoning LLMs, reasoning models find generation more challenging than discrimination and may underperform as generators compared to smaller non-reasoning LLMs. Our work highlights the potential of reasoning models as discriminators in agentic frameworks, far outweighing their capabilities as generators, offering insights into their optimal role within LLM planning infrastructures. 

---
