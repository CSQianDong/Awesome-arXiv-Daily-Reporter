# A Comparative Analysis of Static Word Embeddings for Hungarian 

**Authors**: Máté Gedeon  

**Link**: [PDF](https://arxiv.org/pdf/2505.07809)  

**Abstract**: This paper presents a comprehensive analysis of various static word embeddings for Hungarian, including traditional models such as Word2Vec, FastText, as well as static embeddings derived from BERT-based models using different extraction methods. We evaluate these embeddings on both intrinsic and extrinsic tasks to provide a holistic view of their performance. For intrinsic evaluation, we employ a word analogy task, which assesses the embeddings ability to capture semantic and syntactic relationships. Our results indicate that traditional static embeddings, particularly FastText, excel in this task, achieving high accuracy and mean reciprocal rank (MRR) scores. Among the BERT-based models, the X2Static method for extracting static embeddings demonstrates superior performance compared to decontextualized and aggregate methods, approaching the effectiveness of traditional static embeddings. For extrinsic evaluation, we utilize a bidirectional LSTM model to perform Named Entity Recognition (NER) and Part-of-Speech (POS) tagging tasks. The results reveal that embeddings derived from dynamic models, especially those extracted using the X2Static method, outperform purely static embeddings. Notably, ELMo embeddings achieve the highest accuracy in both NER and POS tagging tasks, underscoring the benefits of contextualized representations even when used in a static form. Our findings highlight the continued relevance of static word embeddings in NLP applications and the potential of advanced extraction methods to enhance the utility of BERT-based models. This piece of research contributes to the understanding of embedding performance in the Hungarian language and provides valuable insights for future developments in the field. The training scripts, evaluation codes, restricted vocabulary, and extracted embeddings will be made publicly available to support further research and reproducibility. 

---
# Learning Dynamics in Continual Pre-Training for Large Language Models 

**Authors**: Xingjin Wang, Howe Tissue, Lu Wang, Linjing Li, Daniel Dajun Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.07796)  

**Abstract**: Continual Pre-Training (CPT) has become a popular and effective method to apply strong foundation models to specific downstream tasks. In this work, we explore the learning dynamics throughout the CPT process for large language models. We specifically focus on how general and downstream domain performance evolves at each training step, with domain performance measured via validation losses. We have observed that the CPT loss curve fundamentally characterizes the transition from one curve to another hidden curve, and could be described by decoupling the effects of distribution shift and learning rate annealing. We derive a CPT scaling law that combines the two factors, enabling the prediction of loss at any (continual) training steps and across learning rate schedules (LRS) in CPT. Our formulation presents a comprehensive understanding of several critical factors in CPT, including loss potential, peak learning rate, training steps, replay ratio, etc. Moreover, our approach can be adapted to customize training hyper-parameters to different CPT goals such as balancing general and domain-specific performance. Extensive experiments demonstrate that our scaling law holds across various CPT datasets and training hyper-parameters. 

---
# Learning from Peers in Reasoning Models 

**Authors**: Tongxu Luo, Wenyu Du, Jiaxi Bi, Stephen Chung, Zhengyang Tang, Hao Yang, Min Zhang, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07787)  

**Abstract**: Large Reasoning Models (LRMs) have the ability to self-correct even when they make mistakes in their reasoning paths. However, our study reveals that when the reasoning process starts with a short but poor beginning, it becomes difficult for the model to recover. We refer to this phenomenon as the "Prefix Dominance Trap". Inspired by psychological findings that peer interaction can promote self-correction without negatively impacting already accurate individuals, we propose **Learning from Peers** (LeaP) to address this phenomenon. Specifically, every tokens, each reasoning path summarizes its intermediate reasoning and shares it with others through a routing mechanism, enabling paths to incorporate peer insights during inference. However, we observe that smaller models sometimes fail to follow summarization and reflection instructions effectively. To address this, we fine-tune them into our **LeaP-T** model series. Experiments on AIME 2024, AIME 2025, AIMO 2025, and GPQA Diamond show that LeaP provides substantial improvements. For instance, QwQ-32B with LeaP achieves nearly 5 absolute points higher than the baseline on average, and surpasses DeepSeek-R1-671B on three math benchmarks with an average gain of 3.3 points. Notably, our fine-tuned LeaP-T-7B matches the performance of DeepSeek-R1-Distill-Qwen-14B on AIME 2024. In-depth analysis reveals LeaP's robust error correction by timely peer insights, showing strong error tolerance and handling varied task difficulty. LeaP marks a milestone by enabling LRMs to collaborate during reasoning. Our code, datasets, and models are available at this https URL . 

---
# Domain Regeneration: How well do LLMs match syntactic properties of text domains? 

**Authors**: Da Ju, Hagen Blix, Adina Williams  

**Link**: [PDF](https://arxiv.org/pdf/2505.07784)  

**Abstract**: Recent improvement in large language model performance have, in all likelihood, been accompanied by improvement in how well they can approximate the distribution of their training data. In this work, we explore the following question: which properties of text domains do LLMs faithfully approximate, and how well do they do so? Applying observational approaches familiar from corpus linguistics, we prompt a commonly used, opensource LLM to regenerate text from two domains of permissively licensed English text which are often contained in LLM training data -- Wikipedia and news text. This regeneration paradigm allows us to investigate whether LLMs can faithfully match the original human text domains in a fairly semantically-controlled setting. We investigate varying levels of syntactic abstraction, from more simple properties like sentence length, and article readability, to more complex and higher order properties such as dependency tag distribution, parse depth, and parse complexity. We find that the majority of the regenerated distributions show a shifted mean, a lower standard deviation, and a reduction of the long tail, as compared to the human originals. 

---
# Must Read: A Systematic Survey of Computational Persuasion 

**Authors**: Nimet Beyza Bozdag, Shuhaib Mehri, Xiaocheng Yang, Hyeonjeong Ha, Zirui Cheng, Esin Durmus, Jiaxuan You, Heng Ji, Gokhan Tur, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2505.07775)  

**Abstract**: Persuasion is a fundamental aspect of communication, influencing decision-making across diverse contexts, from everyday conversations to high-stakes scenarios such as politics, marketing, and law. The rise of conversational AI systems has significantly expanded the scope of persuasion, introducing both opportunities and risks. AI-driven persuasion can be leveraged for beneficial applications, but also poses threats through manipulation and unethical influence. Moreover, AI systems are not only persuaders, but also susceptible to persuasion, making them vulnerable to adversarial attacks and bias reinforcement. Despite rapid advancements in AI-generated persuasive content, our understanding of what makes persuasion effective remains limited due to its inherently subjective and context-dependent nature. In this survey, we provide a comprehensive overview of computational persuasion, structured around three key perspectives: (1) AI as a Persuader, which explores AI-generated persuasive content and its applications; (2) AI as a Persuadee, which examines AI's susceptibility to influence and manipulation; and (3) AI as a Persuasion Judge, which analyzes AI's role in evaluating persuasive strategies, detecting manipulation, and ensuring ethical persuasion. We introduce a taxonomy for computational persuasion research and discuss key challenges, including evaluating persuasiveness, mitigating manipulative persuasion, and developing responsible AI-driven persuasive systems. Our survey outlines future research directions to enhance the safety, fairness, and effectiveness of AI-powered persuasion while addressing the risks posed by increasingly capable language models. 

---
# Spoken Language Understanding on Unseen Tasks With In-Context Learning 

**Authors**: Neeraj Agrawal, Sriram Ganapathy  

**Link**: [PDF](https://arxiv.org/pdf/2505.07731)  

**Abstract**: Spoken language understanding (SLU) tasks involve diverse skills that probe the information extraction, classification and/or generation capabilities of models. In this setting, task-specific training data may not always be available. While traditional task-specific SLU models are unable to cater to such requirements, the speech-text large language models (LLMs) offer a promising alternative with emergent abilities. However, out of-the-box, our evaluations indicate that the zero/few-shot performance of prominent open-source speech-text LLMs on SLU tasks are not up to the mark. In this paper, we introduce a novel approach to robust task-agnostic fine-tuning using randomized class labels. With this proposed fine-tuning, we illustrate that the performance of the speech-text LLMs on an unseen task is significantly improved over standard approaches. Critically, the proposed approach avoids the requirement of task-specific data annotations for enabling new tasks in speech-text LLMs. 

---
# Codifying Character Logic in Role-Playing 

**Authors**: Letian Peng, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07705)  

**Abstract**: This paper introduces Codified Profiles for role-playing, a novel approach that represents character logic as structured, executable functions for behavioral decision-making. Each profile defines a set of functions parse_by_scene(scene) that outputs a list of logic-grounded assertions triggered_statements, using both explicit control structures (e.g., if-then-else) and condition checks like check_condition(scene, question), where each question is a semantically meaningful prompt about the scene (e.g., "Is the character in danger?") discriminated by the role-playing LLM as true, false, or unknown. This explicit representation offers three key advantages over traditional prompt-based profiles, which append character descriptions directly into text prompts: (1) Persistence, by enforcing complete and consistent execution of character logic, rather than relying on the model's implicit reasoning; (2) Updatability, through systematic inspection and revision of behavioral logic, which is difficult to track or debug in prompt-only approaches; (3) Controllable Randomness, by supporting stochastic behavior directly within the logic, enabling fine-grained variability that prompting alone struggles to achieve. To validate these advantages, we introduce a new benchmark constructed from 83 characters and 5,141 scenes curated from Fandom, using NLI-based scoring to compare character responses against ground-truth actions. Our experiments demonstrate the significant benefits of codified profiles in improving persistence, updatability, and behavioral diversity. Notably, by offloading a significant portion of reasoning to preprocessing, codified profiles enable even 1B-parameter models to perform high-quality role-playing, providing a scalable and efficient foundation for local deployment of role-play agents. 

---
# OnPrem.LLM: A Privacy-Conscious Document Intelligence Toolkit 

**Authors**: Arun S. Maiya  

**Link**: [PDF](https://arxiv.org/pdf/2505.07672)  

**Abstract**: We present this http URL, a Python-based toolkit for applying large language models (LLMs) to sensitive, non-public data in offline or restricted environments. The system is designed for privacy-preserving use cases and provides prebuilt pipelines for document processing and storage, retrieval-augmented generation (RAG), information extraction, summarization, classification, and prompt/output processing with minimal configuration. this http URL supports multiple LLM backends -- including this http URL, Ollama, vLLM, and Hugging Face Transformers -- with quantized model support, GPU acceleration, and seamless backend switching. Although designed for fully local execution, this http URL also supports integration with a wide range of cloud LLM providers when permitted, enabling hybrid deployments that balance performance with data control. A no-code web interface extends accessibility to non-technical users. 

---
# Benchmarking Retrieval-Augmented Generation for Chemistry 

**Authors**: Xianrui Zhong, Bowen Jin, Siru Ouyang, Yanzhen Shen, Qiao Jin, Yin Fang, Zhiyong Lu, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.07671)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as a powerful framework for enhancing large language models (LLMs) with external knowledge, particularly in scientific domains that demand specialized and dynamic information. Despite its promise, the application of RAG in the chemistry domain remains underexplored, primarily due to the lack of high-quality, domain-specific corpora and well-curated evaluation benchmarks. In this work, we introduce ChemRAG-Bench, a comprehensive benchmark designed to systematically assess the effectiveness of RAG across a diverse set of chemistry-related tasks. The accompanying chemistry corpus integrates heterogeneous knowledge sources, including scientific literature, the PubChem database, PubMed abstracts, textbooks, and Wikipedia entries. In addition, we present ChemRAG-Toolkit, a modular and extensible RAG toolkit that supports five retrieval algorithms and eight LLMs. Using ChemRAG-Toolkit, we demonstrate that RAG yields a substantial performance gain -- achieving an average relative improvement of 17.4% over direct inference methods. We further conduct in-depth analyses on retriever architectures, corpus selection, and the number of retrieved passages, culminating in practical recommendations to guide future research and deployment of RAG systems in the chemistry domain. The code and data is available at this https URL. 

---
# Using Information Theory to Characterize Prosodic Typology: The Case of Tone, Pitch-Accent and Stress-Accent 

**Authors**: Ethan Gotlieb Wilcox, Cui Ding, Giovanni Acampa, Tiago Pimentel, Alex Warstadt, Tamar I. Regev  

**Link**: [PDF](https://arxiv.org/pdf/2505.07659)  

**Abstract**: This paper argues that the relationship between lexical identity and prosody -- one well-studied parameter of linguistic variation -- can be characterized using information theory. We predict that languages that use prosody to make lexical distinctions should exhibit a higher mutual information between word identity and prosody, compared to languages that don't. We test this hypothesis in the domain of pitch, which is used to make lexical distinctions in tonal languages, like Cantonese. We use a dataset of speakers reading sentences aloud in ten languages across five language families to estimate the mutual information between the text and their pitch curves. We find that, across languages, pitch curves display similar amounts of entropy. However, these curves are easier to predict given their associated text in the tonal languages, compared to pitch- and stress-accent languages, and thus the mutual information is higher in these languages, supporting our hypothesis. Our results support perspectives that view linguistic typology as gradient, rather than categorical. 

---
# JobHop: A Large-Scale Dataset of Career Trajectories 

**Authors**: Iman Johary, Raphael Romero, Alexandru C. Mara, Tijl De Bie  

**Link**: [PDF](https://arxiv.org/pdf/2505.07653)  

**Abstract**: Understanding labor market dynamics is essential for policymakers, employers, and job seekers. However, comprehensive datasets that capture real-world career trajectories are scarce. In this paper, we introduce JobHop, a large-scale public dataset derived from anonymized resumes provided by VDAB, the public employment service in Flanders, Belgium. Utilizing Large Language Models (LLMs), we process unstructured resume data to extract structured career information, which is then mapped to standardized ESCO occupation codes using a multi-label classification model. This results in a rich dataset of over 2.3 million work experiences, extracted from and grouped into more than 391,000 user resumes and mapped to standardized ESCO occupation codes, offering valuable insights into real-world occupational transitions. This dataset enables diverse applications, such as analyzing labor market mobility, job stability, and the effects of career breaks on occupational transitions. It also supports career path prediction and other data-driven decision-making processes. To illustrate its potential, we explore key dataset characteristics, including job distributions, career breaks, and job transitions, demonstrating its value for advancing labor market research. 

---
# Chronocept: Instilling a Sense of Time in Machines 

**Authors**: Krish Goel, Sanskar Pandey, KS Mahadevan, Harsh Kumar, Vishesh Khadaria  

**Link**: [PDF](https://arxiv.org/pdf/2505.07637)  

**Abstract**: Human cognition is deeply intertwined with a sense of time, known as Chronoception. This sense allows us to judge how long facts remain valid and when knowledge becomes outdated. Despite progress in vision, language, and motor control, AI still struggles to reason about temporal validity. We introduce Chronocept, the first benchmark to model temporal validity as a continuous probability distribution over time. Using skew-normal curves fitted along semantically decomposed temporal axes, Chronocept captures nuanced patterns of emergence, decay, and peak relevance. It includes two datasets: Benchmark I (atomic facts) and Benchmark II (multi-sentence passages). Annotations show strong inter-annotator agreement (84% and 89%). Our baselines predict curve parameters - location, scale, and skewness - enabling interpretable, generalizable learning and outperforming classification-based approaches. Chronocept fills a foundational gap in AI's temporal reasoning, supporting applications in knowledge grounding, fact-checking, retrieval-augmented generation (RAG), and proactive agents. Code and data are publicly available. 

---
# Concept-Level Explainability for Auditing & Steering LLM Responses 

**Authors**: Kenza Amara, Rita Sevastjanova, Mennatallah El-Assady  

**Link**: [PDF](https://arxiv.org/pdf/2505.07610)  

**Abstract**: As large language models (LLMs) become widely deployed, concerns about their safety and alignment grow. An approach to steer LLM behavior, such as mitigating biases or defending against jailbreaks, is to identify which parts of a prompt influence specific aspects of the model's output. Token-level attribution methods offer a promising solution, but still struggle in text generation, explaining the presence of each token in the output separately, rather than the underlying semantics of the entire LLM response. We introduce ConceptX, a model-agnostic, concept-level explainability method that identifies the concepts, i.e., semantically rich tokens in the prompt, and assigns them importance based on the outputs' semantic similarity. Unlike current token-level methods, ConceptX also offers to preserve context integrity through in-place token replacements and supports flexible explanation goals, e.g., gender bias. ConceptX enables both auditing, by uncovering sources of bias, and steering, by modifying prompts to shift the sentiment or reduce the harmfulness of LLM responses, without requiring retraining. Across three LLMs, ConceptX outperforms token-level methods like TokenSHAP in both faithfulness and human alignment. Steering tasks boost sentiment shift by 0.252 versus 0.131 for random edits and lower attack success rates from 0.463 to 0.242, outperforming attribution and paraphrasing baselines. While prompt engineering and self-explaining methods sometimes yield safer responses, ConceptX offers a transparent and faithful alternative for improving LLM safety and alignment, demonstrating the practical value of attribution-based explainability in guiding LLM behavior. 

---
# MiMo: Unlocking the Reasoning Potential of Language Model -- From Pretraining to Posttraining 

**Authors**: Xiaomi LLM-Core Team, Bingquan Xia, Bowen Shen, Cici, Dawei Zhu, Di Zhang, Gang Wang, Hailin Zhang, Huaqiu Liu, Jiebao Xiao, Jinhao Dong, Liang Zhao, Peidian Li, Peng Wang, Shihua Yu, Shimao Chen, Weikun Wang, Wenhan Ma, Xiangwei Deng, Yi Huang, Yifan Song, Zihan Jiang, Bowen Ye, Can Cai, Chenhong He, Dong Zhang, Duo Zhang, Guoan Wang, Hao Tian, Haochen Zhao, Heng Qu, Hongshen Xu, Jun Shi, Kainan Bao, QingKai Fang, Kang Zhou, Kangyang Zhou, Lei Li, Menghang Zhu, Nuo Chen, Qiantong Wang, Shaohui Liu, Shicheng Li, Shuhao Gu, Shuhuai Ren, Shuo Liu, Sirui Deng, Weiji Zhuang, Weiwei Lv, Wenyu Yang, Xin Zhang, Xing Yong, Xing Zhang, Xingchen Song, Xinzhe Xu, Xu Wang, Yihan Yan, Yu Tu, Yuanyuan Tian, Yudong Wang, Yue Yu, Zhenru Lin, Zhichao Song, Zihao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2505.07608)  

**Abstract**: We present MiMo-7B, a large language model born for reasoning tasks, with optimization across both pre-training and post-training stages. During pre-training, we enhance the data preprocessing pipeline and employ a three-stage data mixing strategy to strengthen the base model's reasoning potential. MiMo-7B-Base is pre-trained on 25 trillion tokens, with additional Multi-Token Prediction objective for enhanced performance and accelerated inference speed. During post-training, we curate a dataset of 130K verifiable mathematics and programming problems for reinforcement learning, integrating a test-difficulty-driven code-reward scheme to alleviate sparse-reward issues and employing strategic data resampling to stabilize training. Extensive evaluations show that MiMo-7B-Base possesses exceptional reasoning potential, outperforming even much larger 32B models. The final RL-tuned model, MiMo-7B-RL, achieves superior performance on mathematics, code and general reasoning tasks, surpassing the performance of OpenAI o1-mini. The model checkpoints are available at this https URL. 

---
# Characterizing the Investigative Methods of Fictional Detectives with Large Language Models 

**Authors**: Edirlei Soares de Lima, Marco A. Casanova, Bruno Feijó, Antonio L. Furtado  

**Link**: [PDF](https://arxiv.org/pdf/2505.07601)  

**Abstract**: Detective fiction, a genre defined by its complex narrative structures and character-driven storytelling, presents unique challenges for computational narratology, a research field focused on integrating literary theory into automated narrative generation. While traditional literary studies have offered deep insights into the methods and archetypes of fictional detectives, these analyses often focus on a limited number of characters and lack the scalability needed for the extraction of unique traits that can be used to guide narrative generation methods. In this paper, we present an AI-driven approach for systematically characterizing the investigative methods of fictional detectives. Our multi-phase workflow explores the capabilities of 15 Large Language Models (LLMs) to extract, synthesize, and validate distinctive investigative traits of fictional detectives. This approach was tested on a diverse set of seven iconic detectives - Hercule Poirot, Sherlock Holmes, William Murdoch, Columbo, Father Brown, Miss Marple, and Auguste Dupin - capturing the distinctive investigative styles that define each character. The identified traits were validated against existing literary analyses and further tested in a reverse identification phase, achieving an overall accuracy of 91.43%, demonstrating the method's effectiveness in capturing the distinctive investigative approaches of each detective. This work contributes to the broader field of computational narratology by providing a scalable framework for character analysis, with potential applications in AI-driven interactive storytelling and automated narrative generation. 

---
# Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent 

**Authors**: Ziyang Huang, Xiaowei Yuan, Yiming Ju, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07596)  

**Abstract**: Retrieval-augmented generation (RAG) is a common strategy to reduce hallucinations in Large Language Models (LLMs). While reinforcement learning (RL) can enable LLMs to act as search agents by activating retrieval capabilities, existing ones often underutilize their internal knowledge. This can lead to redundant retrievals, potential harmful knowledge conflicts, and increased inference latency. To address these limitations, an efficient and adaptive search agent capable of discerning optimal retrieval timing and synergistically integrating parametric (internal) and retrieved (external) knowledge is in urgent need. This paper introduces the Reinforced Internal-External Knowledge Synergistic Reasoning Agent (IKEA), which could indentify its own knowledge boundary and prioritize the utilization of internal knowledge, resorting to external search only when internal knowledge is deemed insufficient. This is achieved using a novel knowledge-boundary aware reward function and a knowledge-boundary aware training dataset. These are designed for internal-external knowledge synergy oriented RL, incentivizing the model to deliver accurate answers, minimize unnecessary retrievals, and encourage appropriate external searches when its own knowledge is lacking. Evaluations across multiple knowledge reasoning tasks demonstrate that IKEA significantly outperforms baseline methods, reduces retrieval frequency significantly, and exhibits robust generalization capabilities. 

---
# A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models 

**Authors**: Junjie Ye, Caishuang Huang, Zhuohan Chen, Wenjie Fu, Chenyuan Yang, Leyi Yang, Yilong Wu, Peng Wang, Meng Zhou, Xiaolong Yang, Tao Gui, Qi Zhang, Zhongchao Shi, Jianping Fan, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07591)  

**Abstract**: Instruction following evaluates large language models (LLMs) on their ability to generate outputs that adhere to user-defined constraints. However, existing benchmarks often rely on templated constraint prompts, which lack the diversity of real-world usage and limit fine-grained performance assessment. To fill this gap, we propose a multi-dimensional constraint framework encompassing three constraint patterns, four constraint categories, and four difficulty levels. Building on this framework, we develop an automated instruction generation pipeline that performs constraint expansion, conflict detection, and instruction rewriting, yielding 1,200 code-verifiable instruction-following test samples. We evaluate 19 LLMs across seven model families and uncover substantial variation in performance across constraint forms. For instance, average performance drops from 77.67% at Level I to 32.96% at Level IV. Furthermore, we demonstrate the utility of our approach by using it to generate data for reinforcement learning, achieving substantial gains in instruction following without degrading general performance. In-depth analysis indicates that these gains stem primarily from modifications in the model's attention modules parameters, which enhance constraint recognition and adherence. Code and data are available in this https URL. 

---
# SEReDeEP: Hallucination Detection in Retrieval-Augmented Models via Semantic Entropy and Context-Parameter Fusion 

**Authors**: Lei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07528)  

**Abstract**: Retrieval-Augmented Generation (RAG) models frequently encounter hallucination phenomena when integrating external information with internal parametric knowledge. Empirical studies demonstrate that the disequilibrium between external contextual information and internal parametric knowledge constitutes a primary factor in hallucination generation. Existing hallucination detection methodologies predominantly emphasize either the external or internal mechanism in isolation, thereby overlooking their synergistic effects. The recently proposed ReDeEP framework decouples these dual mechanisms, identifying two critical contributors to hallucinations: excessive reliance on parametric knowledge encoded in feed-forward networks (FFN) and insufficient utilization of external information by attention mechanisms (particularly copy heads). ReDeEP quantitatively assesses these factors to detect hallucinations and dynamically modulates the contributions of FFNs and copy heads to attenuate their occurrence. Nevertheless, ReDeEP and numerous other hallucination detection approaches have been employed at logit-level uncertainty estimation or language-level self-consistency evaluation, inadequately address the semantic dimensions of model responses, resulting in inconsistent hallucination assessments in RAG implementations. Building upon ReDeEP's foundation, this paper introduces SEReDeEP, which enhances computational processes through semantic entropy captured via trained linear probes, thereby achieving hallucination assessments that more accurately reflect ground truth evaluations. 

---
# ToolACE-DEV: Self-Improving Tool Learning via Decomposition and EVolution 

**Authors**: Xu Huang, Weiwen Liu, Xingshan Zeng, Yuefeng Huang, Xinlong Hao, Yuxian Wang, Yirong Zeng, Chuhan Wu, Yasheng Wang, Ruiming Tang, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2505.07512)  

**Abstract**: The tool-using capability of large language models (LLMs) enables them to access up-to-date external information and handle complex tasks. Current approaches to enhancing this capability primarily rely on distilling advanced models by data synthesis. However, this method incurs significant costs associated with advanced model usage and often results in data compatibility issues, led by the high discrepancy in the knowledge scope between the advanced model and the target model. To address these challenges, we propose ToolACE-DEV, a self-improving framework for tool learning. First, we decompose the tool-learning objective into sub-tasks that enhance basic tool-making and tool-using abilities. Then, we introduce a self-evolving paradigm that allows lightweight models to self-improve, reducing reliance on advanced LLMs. Extensive experiments validate the effectiveness of our approach across models of varying scales and architectures. 

---
# Translating the Grievance Dictionary: a psychometric evaluation of Dutch, German, and Italian versions 

**Authors**: Isabelle van der Vegt, Bennett Kleinberg, Marilu Miotto, Jonas Festor  

**Link**: [PDF](https://arxiv.org/pdf/2505.07495)  

**Abstract**: This paper introduces and evaluates three translations of the Grievance Dictionary, a psycholinguistic dictionary for the analysis of violent, threatening or grievance-fuelled texts. Considering the relevance of these themes in languages beyond English, we translated the Grievance Dictionary to Dutch, German, and Italian. We describe the process of automated translation supplemented by human annotation. Psychometric analyses are performed, including internal reliability of dictionary categories and correlations with the LIWC dictionary. The Dutch and German translations perform similarly to the original English version, whereas the Italian dictionary shows low reliability for some categories. Finally, we make suggestions for further validation and application of the dictionary, as well as for future dictionary translations following a similar approach. 

---
# Matching Tasks with Industry Groups for Augmenting Commonsense Knowledge 

**Authors**: Rituraj Singh, Sachin Pawar, Girish Palshikar  

**Link**: [PDF](https://arxiv.org/pdf/2505.07440)  

**Abstract**: Commonsense knowledge bases (KB) are a source of specialized knowledge that is widely used to improve machine learning applications. However, even for a large KB such as ConceptNet, capturing explicit knowledge from each industry domain is challenging. For example, only a few samples of general {\em tasks} performed by various industries are available in ConceptNet. Here, a task is a well-defined knowledge-based volitional action to achieve a particular goal. In this paper, we aim to fill this gap and present a weakly-supervised framework to augment commonsense KB with tasks carried out by various industry groups (IG). We attempt to {\em match} each task with one or more suitable IGs by training a neural model to learn task-IG affinity and apply clustering to select the top-k tasks per IG. We extract a total of 2339 triples of the form $\langle IG, is~capable~of, task \rangle$ from two publicly available news datasets for 24 IGs with the precision of 0.86. This validates the reliability of the extracted task-IG pairs that can be directly added to existing KBs. 

---
# Comparative sentiment analysis of public perception: Monkeypox vs. COVID-19 behavioral insights 

**Authors**: Mostafa Mohaimen Akand Faisal, Rabeya Amin Jhuma  

**Link**: [PDF](https://arxiv.org/pdf/2505.07430)  

**Abstract**: The emergence of global health crises, such as COVID-19 and Monkeypox (mpox), has underscored the importance of understanding public sentiment to inform effective public health strategies. This study conducts a comparative sentiment analysis of public perceptions surrounding COVID-19 and mpox by leveraging extensive datasets of 147,475 and 106,638 tweets, respectively. Advanced machine learning models, including Logistic Regression, Naive Bayes, RoBERTa, DistilRoBERTa and XLNet, were applied to perform sentiment classification, with results indicating key trends in public emotion and discourse. The analysis highlights significant differences in public sentiment driven by disease characteristics, media representation, and pandemic fatigue. Through the lens of sentiment polarity and thematic trends, this study offers valuable insights into tailoring public health messaging, mitigating misinformation, and fostering trust during concurrent health crises. The findings contribute to advancing sentiment analysis applications in public health informatics, setting the groundwork for enhanced real-time monitoring and multilingual analysis in future research. 

---
# ViMRHP: A Vietnamese Benchmark Dataset for Multimodal Review Helpfulness Prediction via Human-AI Collaborative Annotation 

**Authors**: Truc Mai-Thanh Nguyen, Dat Minh Nguyen, Son T. Luu, Kiet Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07416)  

**Abstract**: Multimodal Review Helpfulness Prediction (MRHP) is an essential task in recommender systems, particularly in E-commerce platforms. Determining the helpfulness of user-generated reviews enhances user experience and improves consumer decision-making. However, existing datasets focus predominantly on English and Indonesian, resulting in a lack of linguistic diversity, especially for low-resource languages such as Vietnamese. In this paper, we introduce ViMRHP (Vietnamese Multimodal Review Helpfulness Prediction), a large-scale benchmark dataset for MRHP task in Vietnamese. This dataset covers four domains, including 2K products with 46K reviews. Meanwhile, a large-scale dataset requires considerable time and cost. To optimize the annotation process, we leverage AI to assist annotators in constructing the ViMRHP dataset. With AI assistance, annotation time is reduced (90 to 120 seconds per task down to 20 to 40 seconds per task) while maintaining data quality and lowering overall costs by approximately 65%. However, AI-generated annotations still have limitations in complex annotation tasks, which we further examine through a detailed performance analysis. In our experiment on ViMRHP, we evaluate baseline models on human-verified and AI-generated annotations to assess their quality differences. The ViMRHP dataset is publicly available at this https URL 

---
# Computational Fact-Checking of Online Discourse: Scoring scientific accuracy in climate change related news articles 

**Authors**: Tim Wittenborg, Constantin Sebastian Tremel, Markus Stocker, Sören Auer  

**Link**: [PDF](https://arxiv.org/pdf/2505.07409)  

**Abstract**: Democratic societies need reliable information. Misinformation in popular media such as news articles or videos threatens to impair civic discourse. Citizens are, unfortunately, not equipped to verify this content flood consumed daily at increasing rates. This work aims to semi-automatically quantify scientific accuracy of online media. By semantifying media of unknown veracity, their statements can be compared against equally processed trusted sources. We implemented a workflow using LLM-based statement extraction and knowledge graph analysis. Our neurosymbolic system was able to evidently streamline state-of-the-art veracity quantification. Evaluated via expert interviews and a user survey, the tool provides a beneficial veracity indication. This indicator, however, is unable to annotate public media at the required granularity and scale. Further work towards a FAIR (Findable, Accessible, Interoperable, Reusable) ground truth and complementary metrics are required to scientifically support civic discourse. 

---
# QUPID: Quantified Understanding for Enhanced Performance, Insights, and Decisions in Korean Search Engines 

**Authors**: Ohjoon Kwon, Changsu Lee, Jihye Back, Lim Sun Suk, Inho Kang, Donghyeon Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2505.07345)  

**Abstract**: Large language models (LLMs) have been widely used for relevance assessment in information retrieval. However, our study demonstrates that combining two distinct small language models (SLMs) with different architectures can outperform LLMs in this task. Our approach -- QUPID -- integrates a generative SLM with an embedding-based SLM, achieving higher relevance judgment accuracy while reducing computational costs compared to state-of-the-art LLM solutions. This computational efficiency makes QUPID highly scalable for real-world search systems processing millions of queries daily. In experiments across diverse document types, our method demonstrated consistent performance improvements (Cohen's Kappa of 0.646 versus 0.387 for leading LLMs) while offering 60x faster inference times. Furthermore, when integrated into production search pipelines, QUPID improved nDCG@5 scores by 1.9%. These findings underscore how architectural diversity in model combinations can significantly enhance both search relevance and operational efficiency in information retrieval systems. 

---
# Towards Multi-Agent Reasoning Systems for Collaborative Expertise Delegation: An Exploratory Design Study 

**Authors**: Baixuan Xu, Chunyang Li, Weiqi Wang, Wei Fan, Tianshi Zheng, Haochen Shi, Tao Fan, Yangqiu Song, Qiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07313)  

**Abstract**: Designing effective collaboration structure for multi-agent LLM systems to enhance collective reasoning is crucial yet remains under-explored. In this paper, we systematically investigate how collaborative reasoning performance is affected by three key design dimensions: (1) Expertise-Domain Alignment, (2) Collaboration Paradigm (structured workflow vs. diversity-driven integration), and (3) System Scale. Our findings reveal that expertise alignment benefits are highly domain-contingent, proving most effective for contextual reasoning tasks. Furthermore, collaboration focused on integrating diverse knowledge consistently outperforms rigid task decomposition. Finally, we empirically explore the impact of scaling the multi-agent system with expertise specialization and study the computational trade off, highlighting the need for more efficient communication protocol design. This work provides concrete guidelines for configuring specialized multi-agent system and identifies critical architectural trade-offs and bottlenecks for scalable multi-agent reasoning. The code will be made available upon acceptance. 

---
# AttentionInfluence: Adopting Attention Head Influence for Weak-to-Strong Pretraining Data Selection 

**Authors**: Kai Hua, Steven Wu, Ge Zhang, Ke Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07293)  

**Abstract**: Recently, there has been growing interest in collecting reasoning-intensive pretraining data to improve LLMs' complex reasoning ability. Prior approaches typically rely on supervised classifiers to identify such data, which requires labeling by humans or LLMs, often introducing domain-specific biases. Due to the attention heads being crucial to in-context reasoning, we propose AttentionInfluence, a simple yet effective, training-free method without supervision signal. Our approach enables a small pretrained language model to act as a strong data selector through a simple attention head masking operation. Specifically, we identify retrieval heads and compute the loss difference when masking these heads. We apply AttentionInfluence to a 1.3B-parameter dense model to conduct data selection on the SmolLM corpus of 241B tokens, and mix the SmolLM corpus with the selected subset comprising 73B tokens to pretrain a 7B-parameter dense model using 1T training tokens and WSD learning rate scheduling. Our experimental results demonstrate substantial improvements, ranging from 1.4pp to 3.5pp, across several knowledge-intensive and reasoning-heavy benchmarks (i.e., MMLU, MMLU-Pro, AGIEval-en, GSM8K, and HumanEval). This demonstrates an effective weak-to-strong scaling property, with small models improving the final performance of larger models-offering a promising and scalable path for reasoning-centric data selection. 

---
# Semantic Retention and Extreme Compression in LLMs: Can We Have Both? 

**Authors**: Stanislas Laborde, Martin Cousseau, Antoun Yaacoub, Lionel Prevost  

**Link**: [PDF](https://arxiv.org/pdf/2505.07289)  

**Abstract**: The exponential growth in Large Language Model (LLM) deployment has intensified the need for efficient model compression techniques to reduce computational and memory costs. While pruning and quantization have shown promise, their combined potential remains largely unexplored. In this paper, we examine joint compression and how strategically combining pruning and quantization could yield superior performance-to-compression ratios compared to single-method approaches. Recognizing the challenges in accurately assessing LLM performance, we address key limitations of previous evaluation frameworks and introduce the Semantic Retention Compression Rate (SrCr), a novel metric that quantifies the trade-off between model compression and semantic preservation, facilitating the optimization of pruning-quantization configurations. Experiments demonstrate that our recommended combination achieves, on average, a 20% performance increase compared to an equivalent quantization-only model at the same theoretical compression rate. 

---
# On the Robustness of Reward Models for Language Model Alignment 

**Authors**: Jiwoo Hong, Noah Lee, Eunki Kim, Guijin Son, Woojin Chung, Aman Gupta, Shao Tang, James Thorne  

**Link**: [PDF](https://arxiv.org/pdf/2505.07271)  

**Abstract**: The Bradley-Terry (BT) model is widely practiced in reward modeling for reinforcement learning with human feedback (RLHF). Despite its effectiveness, reward models (RMs) trained with BT model loss are prone to over-optimization, losing generalizability to unseen input distributions. In this paper, we study the cause of over-optimization in RM training and its downstream effects on the RLHF procedure, accentuating the importance of distributional robustness of RMs in unseen data. First, we show that the excessive dispersion of hidden state norms is the main source of over-optimization. Then, we propose batch-wise sum-to-zero regularization (BSR) to enforce zero-centered reward sum per batch, constraining the rewards with extreme magnitudes. We assess the impact of BSR in improving robustness in RMs through four scenarios of over-optimization, where BSR consistently manifests better robustness. Subsequently, we compare the plain BT model and BSR on RLHF training and empirically show that robust RMs better align the policy to the gold preference model. Finally, we apply BSR to high-quality data and models, which surpasses state-of-the-art RMs in the 8B scale by adding more than 5% in complex preference prediction tasks. By conducting RLOO training with 8B RMs, AlpacaEval 2.0 reduces generation length by 40% while adding a 7% increase in win rate, further highlighting that robustness in RMs induces robustness in RLHF training. We release the code, data, and models: this https URL. 

---
# No Query, No Access 

**Authors**: Wenqiang Wang, Siyuan Liang, Yangshijie Zhang, Xiaojun Jia, Hao Lin, Xiaochun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.07258)  

**Abstract**: Textual adversarial attacks mislead NLP models, including Large Language Models (LLMs), by subtly modifying text. While effective, existing attacks often require knowledge of the victim model, extensive queries, or access to training data, limiting real-world feasibility. To overcome these constraints, we introduce the \textbf{Victim Data-based Adversarial Attack (VDBA)}, which operates using only victim texts. To prevent access to the victim model, we create a shadow dataset with publicly available pre-trained models and clustering methods as a foundation for developing substitute models. To address the low attack success rate (ASR) due to insufficient information feedback, we propose the hierarchical substitution model design, generating substitute models to mitigate the failure of a single substitute model at the decision boundary.
Concurrently, we use diverse adversarial example generation, employing various attack methods to generate and select the adversarial example with better similarity and attack effectiveness. Experiments on the Emotion and SST5 datasets show that VDBA outperforms state-of-the-art methods, achieving an ASR improvement of 52.08\% while significantly reducing attack queries to 0. More importantly, we discover that VDBA poses a significant threat to LLMs such as Qwen2 and the GPT family, and achieves the highest ASR of 45.99% even without access to the API, confirming that advanced NLP models still face serious security risks. Our codes can be found at this https URL 

---
# SAS-Bench: A Fine-Grained Benchmark for Evaluating Short Answer Scoring with Large Language Models 

**Authors**: Peichao Lai, Kexuan Zhang, Yi Lin, Linyihan Zhang, Feiyang Ye, Jinhao Yan, Yanwei Xu, Conghui He, Yilei Wang, Wentao Zhang, Bin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2505.07247)  

**Abstract**: Subjective Answer Grading (SAG) plays a crucial role in education, standardized testing, and automated assessment systems, particularly for evaluating short-form responses in Short Answer Scoring (SAS). However, existing approaches often produce coarse-grained scores and lack detailed reasoning. Although large language models (LLMs) have demonstrated potential as zero-shot evaluators, they remain susceptible to bias, inconsistencies with human judgment, and limited transparency in scoring decisions. To overcome these limitations, we introduce SAS-Bench, a benchmark specifically designed for LLM-based SAS tasks. SAS-Bench provides fine-grained, step-wise scoring, expert-annotated error categories, and a diverse range of question types derived from real-world subject-specific exams. This benchmark facilitates detailed evaluation of model reasoning processes and explainability. We also release an open-source dataset containing 1,030 questions and 4,109 student responses, each annotated by domain experts. Furthermore, we conduct comprehensive experiments with various LLMs, identifying major challenges in scoring science-related questions and highlighting the effectiveness of few-shot prompting in improving scoring accuracy. Our work offers valuable insights into the development of more robust, fair, and educationally meaningful LLM-based evaluation systems. 

---
# DynamicRAG: Leveraging Outputs of Large Language Model as Feedback for Dynamic Reranking in Retrieval-Augmented Generation 

**Authors**: Jiashuo Sun, Xianrui Zhong, Sizhe Zhou, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.07233)  

**Abstract**: Retrieval-augmented generation (RAG) systems combine large language models (LLMs) with external knowledge retrieval, making them highly effective for knowledge-intensive tasks. A crucial but often under-explored component of these systems is the reranker, which refines retrieved documents to enhance generation quality and explainability. The challenge of selecting the optimal number of documents (k) remains unsolved: too few may omit critical information, while too many introduce noise and inefficiencies. Although recent studies have explored LLM-based rerankers, they primarily leverage internal model knowledge and overlook the rich supervisory signals that LLMs can provide, such as using response quality as feedback for optimizing reranking decisions. In this paper, we propose DynamicRAG, a novel RAG framework where the reranker dynamically adjusts both the order and number of retrieved documents based on the query. We model the reranker as an agent optimized through reinforcement learning (RL), using rewards derived from LLM output quality. Across seven knowledge-intensive datasets, DynamicRAG demonstrates superior performance, achieving state-of-the-art results. The model, data and code are available at this https URL 

---
# Benchmarking Ethical and Safety Risks of Healthcare LLMs in China-Toward Systemic Governance under Healthy China 2030 

**Authors**: Mouxiao Bian, Rongzhao Zhang, Chao Ding, Xinwei Peng, Jie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07205)  

**Abstract**: Large Language Models (LLMs) are poised to transform healthcare under China's Healthy China 2030 initiative, yet they introduce new ethical and patient-safety challenges. We present a novel 12,000-item Q&A benchmark covering 11 ethics and 9 safety dimensions in medical contexts, to quantitatively evaluate these risks. Using this dataset, we assess state-of-the-art Chinese medical LLMs (e.g., Qwen 2.5-32B, DeepSeek), revealing moderate baseline performance (accuracy 42.7% for Qwen 2.5-32B) and significant improvements after fine-tuning on our data (up to 50.8% accuracy). Results show notable gaps in LLM decision-making on ethics and safety scenarios, reflecting insufficient institutional oversight. We then identify systemic governance shortfalls-including the lack of fine-grained ethical audit protocols, slow adaptation by hospital IRBs, and insufficient evaluation tools-that currently hinder safe LLM deployment. Finally, we propose a practical governance framework for healthcare institutions (embedding LLM auditing teams, enacting data ethics guidelines, and implementing safety simulation pipelines) to proactively manage LLM risks. Our study highlights the urgent need for robust LLM governance in Chinese healthcare, aligning AI innovation with patient safety and ethical standards. 

---
# On the Cost and Benefits of Training Context with Utterance or Full Conversation Training: A Comparative Stud 

**Authors**: Hyouin Liu, Zhikuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07202)  

**Abstract**: Modern TTS systems designed for conversations achieve high-quality utterances but often remain inaccessible publicly. Are existing open-source architectures inadequate, or are current training techniques insufficient? This paper investigates prominent models and their underlying behaviors regarding conversational context. Using 20 GPU-hours on an NVIDIA H100, we empirically examine two approaches: context-based utterance-level training versus full conversation training. Results demonstrate that context-based utterance training achieves superior MOS scores (4.3/5.0 vs 3.7/5.0) and reduces training time by 37%, while full conversation approaches suffer from speaker similarity hallucination issues. These findings provide practical guidelines for conversational TTS development, favoring utterance-level training with contextual conditioning for both resource efficiency and output quality. 

---
# Structural Entropy Guided Agent for Detecting and Repairing Knowledge Deficiencies in LLMs 

**Authors**: Yifan Wei, Xiaoyan Yu, Tengfei Pan, Angsheng Li, Li Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.07184)  

**Abstract**: Large language models (LLMs) have achieved unprecedented performance by leveraging vast pretraining corpora, yet their performance remains suboptimal in knowledge-intensive domains such as medicine and scientific research, where high factual precision is required. While synthetic data provides a promising avenue for augmenting domain knowledge, existing methods frequently generate redundant samples that do not align with the model's true knowledge gaps. To overcome this limitation, we propose a novel Structural Entropy-guided Knowledge Navigator (SENATOR) framework that addresses the intrinsic knowledge deficiencies of LLMs. Our approach employs the Structure Entropy (SE) metric to quantify uncertainty along knowledge graph paths and leverages Monte Carlo Tree Search (MCTS) to selectively explore regions where the model lacks domain-specific knowledge. Guided by these insights, the framework generates targeted synthetic data for supervised fine-tuning, enabling continuous self-improvement. Experimental results on LLaMA-3 and Qwen2 across multiple domain-specific benchmarks show that SENATOR effectively detects and repairs knowledge deficiencies, achieving notable performance improvements. The code and data for our methods and experiments are available at this https URL. 

---
# KDH-MLTC: Knowledge Distillation for Healthcare Multi-Label Text Classification 

**Authors**: Hajar Sakai, Sarah S. Lam  

**Link**: [PDF](https://arxiv.org/pdf/2505.07162)  

**Abstract**: The increasing volume of healthcare textual data requires computationally efficient, yet highly accurate classification approaches able to handle the nuanced and complex nature of medical terminology. This research presents Knowledge Distillation for Healthcare Multi-Label Text Classification (KDH-MLTC), a framework leveraging model compression and Large Language Models (LLMs). The proposed approach addresses conventional healthcare Multi-Label Text Classification (MLTC) challenges by integrating knowledge distillation and sequential fine-tuning, subsequently optimized through Particle Swarm Optimization (PSO) for hyperparameter tuning. KDH-MLTC transfers knowledge from a more complex teacher LLM (i.e., BERT) to a lighter student LLM (i.e., DistilBERT) through sequential training adapted to MLTC that preserves the teacher's learned information while significantly reducing computational requirements. As a result, the classification is enabled to be conducted locally, making it suitable for healthcare textual data characterized by sensitivity and, therefore, ensuring HIPAA compliance. The experiments conducted on three medical literature datasets of different sizes, sampled from the Hallmark of Cancer (HoC) dataset, demonstrate that KDH-MLTC achieves superior performance compared to existing approaches, particularly for the largest dataset, reaching an F1 score of 82.70%. Additionally, statistical validation and an ablation study are carried out, proving the robustness of KDH-MLTC. Furthermore, the PSO-based hyperparameter optimization process allowed the identification of optimal configurations. The proposed approach contributes to healthcare text classification research, balancing efficiency requirements in resource-constrained healthcare settings with satisfactory accuracy demands. 

---
# Towards Actionable Pedagogical Feedback: A Multi-Perspective Analysis of Mathematics Teaching and Tutoring Dialogue 

**Authors**: Jannatun Naim, Jie Cao, Fareen Tasneem, Jennifer Jacobs, Brent Milne, James Martin, Tamara Sumner  

**Link**: [PDF](https://arxiv.org/pdf/2505.07161)  

**Abstract**: Effective feedback is essential for refining instructional practices in mathematics education, and researchers often turn to advanced natural language processing (NLP) models to analyze classroom dialogues from multiple perspectives. However, utterance-level discourse analysis encounters two primary challenges: (1) multifunctionality, where a single utterance may serve multiple purposes that a single tag cannot capture, and (2) the exclusion of many utterances from domain-specific discourse move classifications, leading to their omission in feedback. To address these challenges, we proposed a multi-perspective discourse analysis that integrates domain-specific talk moves with dialogue act (using the flattened multi-functional SWBD-MASL schema with 43 tags) and discourse relation (applying Segmented Discourse Representation Theory with 16 relations). Our top-down analysis framework enables a comprehensive understanding of utterances that contain talk moves, as well as utterances that do not contain talk moves. This is applied to two mathematics education datasets: TalkMoves (teaching) and SAGA22 (tutoring). Through distributional unigram analysis, sequential talk move analysis, and multi-view deep dive, we discovered meaningful discourse patterns, and revealed the vital role of utterances without talk moves, demonstrating that these utterances, far from being mere fillers, serve crucial functions in guiding, acknowledging, and structuring classroom discourse. These insights underscore the importance of incorporating discourse relations and dialogue acts into AI-assisted education systems to enhance feedback and create more responsive learning environments. Our framework may prove helpful for providing human educator feedback, but also aiding in the development of AI agents that can effectively emulate the roles of both educators and students. 

---
# HAMLET: Healthcare-focused Adaptive Multilingual Learning Embedding-based Topic Modeling 

**Authors**: Hajar Sakai, Sarah S. Lam  

**Link**: [PDF](https://arxiv.org/pdf/2505.07157)  

**Abstract**: Traditional topic models often struggle with contextual nuances and fail to adequately handle polysemy and rare words. This limitation typically results in topics that lack coherence and quality. Large Language Models (LLMs) can mitigate this issue by generating an initial set of topics. However, these raw topics frequently lack refinement and representativeness, which leads to redundancy without lexical similarity and reduced interpretability. This paper introduces HAMLET, a graph-driven architecture for cross-lingual healthcare topic modeling that uses LLMs. The proposed approach leverages neural-enhanced semantic fusion to refine the embeddings of topics generated by the LLM. Instead of relying solely on statistical co-occurrence or human interpretation to extract topics from a document corpus, this method introduces a topic embedding refinement that uses Bidirectional Encoder Representations from Transformers (BERT) and Graph Neural Networks (GNN). After topic generation, a hybrid technique that involves BERT and Sentence-BERT (SBERT) is employed for embedding. The topic representations are further refined using a GNN, which establishes connections between documents, topics, words, similar topics, and similar words. A novel method is introduced to compute similarities. Consequently, the topic embeddings are refined, and the top k topics are extracted. Experiments were conducted using two healthcare datasets, one in English and one in French, from which six sets were derived. The results demonstrate the effectiveness of HAMLET. 

---
# Convert Language Model into a Value-based Strategic Planner 

**Authors**: Xiaoyu Wang, Yue Zhao, Qingqing Gu, Zhonglin Jiang, Xiaokai Chen, Yong Chen, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.06987)  

**Abstract**: Emotional support conversation (ESC) aims to alleviate the emotional distress of individuals through effective conversations. Although large language models (LLMs) have obtained remarkable progress on ESC, most of these studies might not define the diagram from the state model perspective, therefore providing a suboptimal solution for long-term satisfaction. To address such an issue, we leverage the Q-learning on LLMs, and propose a framework called straQ*. Our framework allows a plug-and-play LLM to bootstrap the planning during ESC, determine the optimal strategy based on long-term returns, and finally guide the LLM to response. Substantial experiments on ESC datasets suggest that straQ* outperforms many baselines, including direct inference, self-refine, chain of thought, finetuning, and finite state machines. 

---
# CNN-based Image Models Verify a Hypothesis that The Writers of Cuneiform Texts Improved Their Writing Skills When Studying at the Age of Hittite Empire 

**Authors**: Daichi Kohmoto, Katsutoshi Fukuda, Daisuke Yoshida, Takafumi Matsui, Sachihiro Omura  

**Link**: [PDF](https://arxiv.org/pdf/2505.06974)  

**Abstract**: A cuneiform tablet KBo 23.1 ++/KUB 30.38, which is known to represent a text of Kizzuwatna rituals, was written by two writers with almost identical content in two iterations. Unlike other cuneiform tablets that contained information such as myths, essays, or business records, the reason why ancient people left such tablets for posterity remains unclear. To study this problem, we develop a new methodology by analyzing images of a tablet quantitatively using CNN (Convolutional Neural Network)-based image models, without segmenting cuneiforms one-by-one. Our data-driven methodology implies that the writer writing the first half was a `teacher' and the other writer was a `student' who was training his skills of writing cuneiforms. This result has not been reached by classical linguistics. We also discuss related conclusions and possible further directions for applying our method and its generalizations. 

---
# The Distracting Effect: Understanding Irrelevant Passages in RAG 

**Authors**: Chen Amiraz, Florin Cuconasu, Simone Filice, Zohar Karnin  

**Link**: [PDF](https://arxiv.org/pdf/2505.06914)  

**Abstract**: A well-known issue with Retrieval Augmented Generation (RAG) is that retrieved passages that are irrelevant to the query sometimes distract the answer-generating LLM, causing it to provide an incorrect response. In this paper, we shed light on this core issue and formulate the distracting effect of a passage w.r.t. a query (and an LLM). We provide a quantifiable measure of the distracting effect of a passage and demonstrate its robustness across LLMs.
Our research introduces novel methods for identifying and using hard distracting passages to improve RAG systems. By fine-tuning LLMs with these carefully selected distracting passages, we achieve up to a 7.5% increase in answering accuracy compared to counterparts fine-tuned on conventional RAG datasets. Our contribution is two-fold: first, we move beyond the simple binary classification of irrelevant passages as either completely unrelated vs. distracting, and second, we develop and analyze multiple methods for finding hard distracting passages. To our knowledge, no other research has provided such a comprehensive framework for identifying and utilizing hard distracting passages. 

---
# EcoLANG: Efficient and Effective Agent Communication Language Induction for Social Simulation 

**Authors**: Xinyi Mou, Chen Qian, Wei Liu, Xuanjing Huang, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.06904)  

**Abstract**: Large language models (LLMs) have demonstrated an impressive ability to role-play humans and replicate complex social dynamics. While large-scale social simulations are gaining increasing attention, they still face significant challenges, particularly regarding high time and computation costs. Existing solutions, such as distributed mechanisms or hybrid agent-based model (ABM) integrations, either fail to address inference costs or compromise accuracy and generalizability. To this end, we propose EcoLANG: Efficient and Effective Agent Communication Language Induction for Social Simulation. EcoLANG operates in two stages: (1) language evolution, where we filter synonymous words and optimize sentence-level rules through natural selection, and (2) language utilization, where agents in social simulations communicate using the evolved language. Experimental results demonstrate that EcoLANG reduces token consumption by over 20%, enhancing efficiency without sacrificing simulation accuracy. 

---
# IM-BERT: Enhancing Robustness of BERT through the Implicit Euler Method 

**Authors**: Mihyeon Kim, Juhyoung Park, Youngbin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.06889)  

**Abstract**: Pre-trained Language Models (PLMs) have achieved remarkable performance on diverse NLP tasks through pre-training and fine-tuning. However, fine-tuning the model with a large number of parameters on limited downstream datasets often leads to vulnerability to adversarial attacks, causing overfitting of the model on standard datasets.
To address these issues, we propose IM-BERT from the perspective of a dynamic system by conceptualizing a layer of BERT as a solution of Ordinary Differential Equations (ODEs). Under the situation of initial value perturbation, we analyze the numerical stability of two main numerical ODE solvers: the explicit and implicit Euler approaches.
Based on these analyses, we introduce a numerically robust IM-connection incorporating BERT's layers. This strategy enhances the robustness of PLMs against adversarial attacks, even in low-resource scenarios, without introducing additional parameters or adversarial training strategies.
Experimental results on the adversarial GLUE (AdvGLUE) dataset validate the robustness of IM-BERT under various conditions. Compared to the original BERT, IM-BERT exhibits a performance improvement of approximately 8.3\%p on the AdvGLUE dataset. Furthermore, in low-resource scenarios, IM-BERT outperforms BERT by achieving 5.9\%p higher accuracy. 

---
# A Split-then-Join Approach to Abstractive Summarization for Very Long Documents in a Low Resource Setting 

**Authors**: Lhuqita Fazry  

**Link**: [PDF](https://arxiv.org/pdf/2505.06862)  

**Abstract**: $\texttt{BIGBIRD-PEGASUS}$ model achieves $\textit{state-of-the-art}$ on abstractive text summarization for long documents. However it's capacity still limited to maximum of $4,096$ tokens, thus caused performance degradation on summarization for very long documents. Common method to deal with the issue is to truncate the documents. In this reasearch, we'll use different approach. We'll use the pretrained $\texttt{BIGBIRD-PEGASUS}$ model by fine tuned the model on other domain dataset. First, we filter out all documents which length less than $20,000$ tokens to focus on very long documents. To prevent domain shifting problem and overfitting on transfer learning due to small dataset, we augment the dataset by splitting document-summary training pair into parts, to fit the document into $4,096$ tokens. Source code available on $\href{this https URL}{this https URL}$. 

---
# Utilizing LLMs to Investigate the Disputed Role of Evidence in Electronic Cigarette Health Policy Formation in Australia and the UK 

**Authors**: Damian Curran, Brian Chapman, Mike Conway  

**Link**: [PDF](https://arxiv.org/pdf/2505.06782)  

**Abstract**: Australia and the UK have developed contrasting approaches to the regulation of electronic cigarettes, with - broadly speaking - Australia adopting a relatively restrictive approach and the UK adopting a more permissive approach. Notably, these divergent policies were developed from the same broad evidence base. In this paper, to investigate differences in how the two jurisdictions manage and present evidence, we developed and evaluated a Large Language Model-based sentence classifier to perform automated analyses of electronic cigarette-related policy documents drawn from official Australian and UK legislative processes (109 documents in total). Specifically, we utilized GPT-4 to automatically classify sentences based on whether they contained claims that e-cigarettes were broadly helpful or harmful for public health. Our LLM-based classifier achieved an F-score of 0.9. Further, when applying the classifier to our entire sentence-level corpus, we found that Australian legislative documents show a much higher proportion of harmful statements, and a lower proportion of helpful statements compared to the expected values, with the opposite holding for the UK. In conclusion, this work utilized an LLM-based approach to provide evidence to support the contention that - drawing on the same evidence base - Australian ENDS-related policy documents emphasize the harms associated with ENDS products and UK policy documents emphasize the benefits. Further, our approach provides a starting point for using LLM-based methods to investigate the complex relationship between evidence and health policy formation. 

---
# Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free 

**Authors**: Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.06708)  

**Abstract**: Gating mechanisms have been widely utilized, from early models like LSTMs and Highway Networks to recent state space models, linear attention, and also softmax attention. Yet, existing literature rarely examines the specific effects of gating. In this work, we conduct comprehensive experiments to systematically investigate gating-augmented softmax attention variants. Specifically, we perform a comprehensive comparison over 30 variants of 15B Mixture-of-Experts (MoE) models and 1.7B dense models trained on a 3.5 trillion token dataset. Our central finding is that a simple modification-applying a head-specific sigmoid gate after the Scaled Dot-Product Attention (SDPA)-consistently improves performance. This modification also enhances training stability, tolerates larger learning rates, and improves scaling properties. By comparing various gating positions and computational variants, we attribute this effectiveness to two key factors: (1) introducing non-linearity upon the low-rank mapping in the softmax attention, and (2) applying query-dependent sparse gating scores to modulate the SDPA output. Notably, we find this sparse gating mechanism mitigates 'attention sink' and enhances long-context extrapolation performance, and we also release related $\href{this https URL}{codes}$ and $\href{this https URL}{models}$ to facilitate future research. 

---
# From Rankings to Insights: Evaluation Should Shift Focus from Leaderboard to Feedback 

**Authors**: Zongqi Wang, Tianle Gu, Chen Gong, Xin Tian, Siqi Bao, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06698)  

**Abstract**: Automatic evaluation benchmarks such as MT-Bench, Arena-Hard, and Auto-Arena are seeing growing adoption for the evaluation of Large Language Models (LLMs). Existing research has primarily focused on approximating human-based model rankings using limited data and LLM-as-a-Judge. However, the fundamental premise of these studies, which attempts to replicate human rankings, is flawed. Specifically, these benchmarks typically offer only overall scores, limiting their utility to leaderboard rankings, rather than providing feedback that can guide model optimization and support model profiling. Therefore, we advocate for an evaluation paradigm shift from approximating human-based model rankings to providing feedback with analytical value. To this end, we introduce Feedbacker, an evaluation framework that provides comprehensive and fine-grained results, thereby enabling thorough identification of a model's specific strengths and weaknesses. Such feedback not only supports the targeted optimization of the model but also enhances the understanding of its behavior. Feedbacker comprises three key components: an extensible tree-based query taxonomy builder, an automated query synthesis scheme, and a suite of visualization and analysis tools. Furthermore, we propose a novel LLM-as-a-Judge method: PC2 (Pre-Comparison-derived Criteria) pointwise evaluation. This method derives evaluation criteria by pre-comparing the differences between several auxiliary responses, achieving the accuracy of pairwise evaluation while maintaining the time complexity of pointwise evaluation. Finally, leveraging the evaluation results of 17 mainstream LLMs, we demonstrate the usage of Feedbacker and highlight its effectiveness and potential. Our homepage project is available at this https URL. 

---
# Enhancing BERTopic with Intermediate Layer Representations 

**Authors**: Dominik Koterwa, Maciej Świtała  

**Link**: [PDF](https://arxiv.org/pdf/2505.06696)  

**Abstract**: BERTopic is a topic modeling algorithm that leverages transformer-based embeddings to create dense clusters, enabling the estimation of topic structures and the extraction of valuable insights from a corpus of documents. This approach allows users to efficiently process large-scale text data and gain meaningful insights into its structure. While BERTopic is a powerful tool, embedding preparation can vary, including extracting representations from intermediate model layers and applying transformations to these embeddings. In this study, we evaluate 18 different embedding representations and present findings based on experiments conducted on three diverse datasets. To assess the algorithm's performance, we report topic coherence and topic diversity metrics across all experiments. Our results demonstrate that, for each dataset, it is possible to find an embedding configuration that performs better than the default setting of BERTopic. Additionally, we investigate the influence of stop words on different embedding configurations. 

---
# TS-SUPERB: A Target Speech Processing Benchmark for Speech Self-Supervised Learning Models 

**Authors**: Junyi Peng, Takanori Ashihara, Marc Delcroix, Tsubasa Ochiai, Oldrich Plchot, Shoko Araki, Jan Černocký  

**Link**: [PDF](https://arxiv.org/pdf/2505.06660)  

**Abstract**: Self-supervised learning (SSL) models have significantly advanced speech processing tasks, and several benchmarks have been proposed to validate their effectiveness. However, previous benchmarks have primarily focused on single-speaker scenarios, with less exploration of target-speaker tasks in noisy, multi-talker conditions -- a more challenging yet practical case. In this paper, we introduce the Target-Speaker Speech Processing Universal Performance Benchmark (TS-SUPERB), which includes four widely recognized target-speaker processing tasks that require identifying the target speaker and extracting information from the speech mixture. In our benchmark, the speaker embedding extracted from enrollment speech is used as a clue to condition downstream models. The benchmark result reveals the importance of evaluating SSL models in target speaker scenarios, demonstrating that performance cannot be easily inferred from related single-speaker tasks. Moreover, by using a unified SSL-based target speech encoder, consisting of a speaker encoder and an extractor module, we also investigate joint optimization across TS tasks to leverage mutual information and demonstrate its effectiveness. 

---
# Attention Is Not All You Need: The Importance of Feedforward Networks in Transformer Models 

**Authors**: Isaac Gerber  

**Link**: [PDF](https://arxiv.org/pdf/2505.06633)  

**Abstract**: Decoder-only transformer networks have become incredibly popular for language modeling tasks. State-of-the-art models can have over a hundred transformer blocks, containing billions of trainable parameters, and are trained on trillions of tokens of text. Each transformer block typically consists of a multi-head attention (MHA) mechanism and a two-layer fully connected feedforward network (FFN). In this paper, we examine the importance of the FFN during the model pre-training process through a series of experiments, confirming that the FFN is important to model performance. Furthermore, we show that models using a transformer block configuration with three-layer FFNs with fewer such blocks outperform the standard two-layer configuration delivering lower training loss with fewer total parameters in less time. 

---
# Dynamic Domain Information Modulation Algorithm for Multi-domain Sentiment Analysis 

**Authors**: Chunyi Yue, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.06630)  

**Abstract**: Multi-domain sentiment classification aims to mitigate poor performance models due to the scarcity of labeled data in a single domain, by utilizing data labeled from various domains. A series of models that jointly train domain classifiers and sentiment classifiers have demonstrated their advantages, because domain classification helps generate necessary information for sentiment classification. Intuitively, the importance of sentiment classification tasks is the same in all domains for multi-domain sentiment classification; but domain classification tasks are different because the impact of domain information on sentiment classification varies across different fields; this can be controlled through adjustable weights or hyper parameters. However, as the number of domains increases, existing hyperparameter optimization algorithms may face the following challenges: (1) tremendous demand for computing resources, (2) convergence problems, and (3) high algorithm complexity. To efficiently generate the domain information required for sentiment classification in each domain, we propose a dynamic information modulation algorithm. Specifically, the model training process is divided into two stages. In the first stage, a shared hyperparameter, which would control the proportion of domain classification tasks across all fields, is determined. In the second stage, we introduce a novel domain-aware modulation algorithm to adjust the domain information contained in the input text, which is then calculated based on a gradient-based and loss-based method. In summary, experimental results on a public sentiment analysis dataset containing 16 domains prove the superiority of the proposed method. 

---
# The Efficiency of Pre-training with Objective Masking in Pseudo Labeling for Semi-Supervised Text Classification 

**Authors**: Arezoo Hatefi, Xuan-Son Vu, Monowar Bhuyan, Frank Drewes  

**Link**: [PDF](https://arxiv.org/pdf/2505.06624)  

**Abstract**: We extend and study a semi-supervised model for text classification proposed earlier by Hatefi et al. for classification tasks in which document classes are described by a small number of gold-labeled examples, while the majority of training examples is unlabeled. The model leverages the teacher-student architecture of Meta Pseudo Labels in which a ''teacher'' generates labels for originally unlabeled training data to train the ''student'' and updates its own model iteratively based on the performance of the student on the gold-labeled portion of the data. We extend the original model of Hatefi et al. by an unsupervised pre-training phase based on objective masking, and conduct in-depth performance evaluations of the original model, our extension, and various independent baselines. Experiments are performed using three different datasets in two different languages (English and Swedish). 

---
# Boosting Neural Language Inference via Cascaded Interactive Reasoning 

**Authors**: Min Li, Chun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06607)  

**Abstract**: Natural Language Inference (NLI) focuses on ascertaining the logical relationship (entailment, contradiction, or neutral) between a given premise and hypothesis. This task presents significant challenges due to inherent linguistic features such as diverse phrasing, semantic complexity, and contextual nuances. While Pre-trained Language Models (PLMs) built upon the Transformer architecture have yielded substantial advancements in NLI, prevailing methods predominantly utilize representations from the terminal layer. This reliance on final-layer outputs may overlook valuable information encoded in intermediate layers, potentially limiting the capacity to model intricate semantic interactions effectively. Addressing this gap, we introduce the Cascaded Interactive Reasoning Network (CIRN), a novel architecture designed for deeper semantic comprehension in NLI. CIRN implements a hierarchical feature extraction strategy across multiple network depths, operating within an interactive space where cross-sentence information is continuously integrated. This mechanism aims to mimic a process of progressive reasoning, transitioning from surface-level feature matching to uncovering more profound logical and semantic connections between the premise and hypothesis. By systematically mining latent semantic relationships at various representational levels, CIRN facilitates a more thorough understanding of the input pair. Comprehensive evaluations conducted on several standard NLI benchmark datasets reveal consistent performance gains achieved by CIRN over competitive baseline approaches, demonstrating the efficacy of leveraging multi-level interactive features for complex relational reasoning. 

---
# Using External knowledge to Enhanced PLM for Semantic Matching 

**Authors**: Min Li, Chun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06605)  

**Abstract**: Modeling semantic relevance has always been a challenging and critical task in natural language processing. In recent years, with the emergence of massive amounts of annotated data, it has become feasible to train complex models, such as neural network-based reasoning models. These models have shown excellent performance in practical applications and have achieved the current state-ofthe-art performance. However, even with such large-scale annotated data, we still need to think: Can machines learn all the knowledge necessary to perform semantic relevance detection tasks based on this data alone? If not, how can neural network-based models incorporate external knowledge into themselves, and how can relevance detection models be constructed to make full use of external knowledge? In this paper, we use external knowledge to enhance the pre-trained semantic relevance discrimination model. Experimental results on 10 public datasets show that our method achieves consistent improvements in performance compared to the baseline model. 

---
# Bridging the Gap: An Intermediate Language for Enhanced and Cost-Effective Grapheme-to-Phoneme Conversion with Homographs with Multiple Pronunciations Disambiguation 

**Authors**: Abbas Bertina, Shahab Beirami, Hossein Biniazian, Elham Esmaeilnia, Soheil Shahi, Mahdi Pirnia  

**Link**: [PDF](https://arxiv.org/pdf/2505.06599)  

**Abstract**: Grapheme-to-phoneme (G2P) conversion for Persian presents unique challenges due to its complex phonological features, particularly homographs and Ezafe, which exist in formal and informal language contexts. This paper introduces an intermediate language specifically designed for Persian language processing that addresses these challenges through a multi-faceted approach. Our methodology combines two key components: Large Language Model (LLM) prompting techniques and a specialized sequence-to-sequence machine transliteration architecture. We developed and implemented a systematic approach for constructing a comprehensive lexical database for homographs with multiple pronunciations disambiguation often termed polyphones, utilizing formal concept analysis for semantic differentiation. We train our model using two distinct datasets: the LLM-generated dataset for formal and informal Persian and the B-Plus podcasts for informal language variants. The experimental results demonstrate superior performance compared to existing state-of-the-art approaches, particularly in handling the complexities of Persian phoneme conversion. Our model significantly improves Phoneme Error Rate (PER) metrics, establishing a new benchmark for Persian G2P conversion accuracy. This work contributes to the growing research in low-resource language processing and provides a robust solution for Persian text-to-speech systems and demonstrating its applicability beyond Persian. Specifically, the approach can extend to languages with rich homographic phenomena such as Chinese and Arabic 

---
# Integrating Video and Text: A Balanced Approach to Multimodal Summary Generation and Evaluation 

**Authors**: Galann Pennec, Zhengyuan Liu, Nicholas Asher, Philippe Muller, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.06594)  

**Abstract**: Vision-Language Models (VLMs) often struggle to balance visual and textual information when summarizing complex multimodal inputs, such as entire TV show episodes. In this paper, we propose a zero-shot video-to-text summarization approach that builds its own screenplay representation of an episode, effectively integrating key video moments, dialogue, and character information into a unified document. Unlike previous approaches, we simultaneously generate screenplays and name the characters in zero-shot, using only the audio, video, and transcripts as input. Additionally, we highlight that existing summarization metrics can fail to assess the multimodal content in summaries. To address this, we introduce MFactSum, a multimodal metric that evaluates summaries with respect to both vision and text modalities. Using MFactSum, we evaluate our screenplay summaries on the SummScreen3D dataset, demonstrating superiority against state-of-the-art VLMs such as Gemini 1.5 by generating summaries containing 20% more relevant visual information while requiring 75% less of the video as input. 

---
# Evaluating LLM-Generated Q&A Test: a Student-Centered Study 

**Authors**: Anna Wróblewska, Bartosz Grabek, Jakub Świstak, Daniel Dan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06591)  

**Abstract**: This research prepares an automatic pipeline for generating reliable question-answer (Q&A) tests using AI chatbots. We automatically generated a GPT-4o-mini-based Q&A test for a Natural Language Processing course and evaluated its psychometric and perceived-quality metrics with students and experts. A mixed-format IRT analysis showed that the generated items exhibit strong discrimination and appropriate difficulty, while student and expert star ratings reflect high overall quality. A uniform DIF check identified two items for review. These findings demonstrate that LLM-generated assessments can match human-authored tests in psychometric performance and user satisfaction, illustrating a scalable approach to AI-assisted assessment development. 

---
# MacRAG: Compress, Slice, and Scale-up for Multi-Scale Adaptive Context RAG 

**Authors**: Woosang Lim, Zekun Li, Gyuwan Kim, Sungyoung Ji, HyeonJung Kim, Kyuri Choi, Jin Hyuk Lim, Kyungpyo Park, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06569)  

**Abstract**: Long-context (LC) Large Language Models (LLMs) combined with Retrieval-Augmented Generation (RAG) hold strong potential for complex multi-hop and large-document tasks. However, existing RAG systems often suffer from imprecise retrieval, incomplete context coverage under constrained context windows, and fragmented information caused by suboptimal context construction. We introduce Multi-scale Adaptive Context RAG (MacRAG), a hierarchical retrieval framework that compresses and partitions documents into coarse-to-fine granularities, then adaptively merges relevant contexts through chunk- and document-level expansions in real time. By starting from the finest-level retrieval and progressively incorporating higher-level and broader context, MacRAG constructs effective query-specific long contexts, optimizing both precision and coverage. Evaluations on the challenging LongBench expansions of HotpotQA, 2WikiMultihopQA, and Musique confirm that MacRAG consistently surpasses baseline RAG pipelines on single- and multi-step generation with Llama-3.1-8B, Gemini-1.5-pro, and GPT-4o. Our results establish MacRAG as an efficient, scalable solution for real-world long-context, multi-hop reasoning. Our code is available at this https URL. 

---
# References Indeed Matter? Reference-Free Preference Optimization for Conversational Query Reformulation 

**Authors**: Doyoung Kim, Youngjun Lee, Joeun Kim, Jihwan Bang, Hwanjun Song, Susik Yoon, Jae-Gil Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.06552)  

**Abstract**: Conversational query reformulation (CQR) has become indispensable for improving retrieval in dialogue-based applications. However, existing approaches typically rely on reference passages for optimization, which are impractical to acquire in real-world scenarios. To address this limitation, we introduce a novel reference-free preference optimization framework DualReform that generates pseudo reference passages from commonly-encountered conversational datasets containing only queries and responses. DualReform attains this goal through two key innovations: (1) response-based inference, where responses serve as proxies to infer pseudo reference passages, and (2) response refinement via the dual-role of CQR, where a CQR model refines responses based on the shared objectives between response refinement and CQR. Despite not relying on reference passages, DualReform achieves 96.9--99.1% of the retrieval accuracy attainable only with reference passages and surpasses the state-of-the-art method by up to 31.6%. 

---
# REFINE-AF: A Task-Agnostic Framework to Align Language Models via Self-Generated Instructions using Reinforcement Learning from Automated Feedback 

**Authors**: Aniruddha Roy, Pretam Ray, Abhilash Nandy, Somak Aditya, Pawan Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2505.06548)  

**Abstract**: Instruction-based Large Language Models (LLMs) have proven effective in numerous few-shot or zero-shot Natural Language Processing (NLP) tasks. However, creating human-annotated instruction data is time-consuming, expensive, and often limited in quantity and task diversity. Previous research endeavors have attempted to address this challenge by proposing frameworks capable of generating instructions in a semi-automated and task-agnostic manner directly from the model itself. Many of these efforts have relied on large API-only parameter-based models such as GPT-3.5 (175B), which are expensive, and subject to limits on a number of queries. This paper explores the performance of three open-source small LLMs such as LLaMA 2-7B, LLama 2-13B, and Mistral 7B, using a semi-automated framework, thereby reducing human intervention, effort, and cost required to generate an instruction dataset for fine-tuning LLMs. Furthermore, we demonstrate that incorporating a Reinforcement Learning (RL) based training algorithm into this LLMs-based framework leads to further enhancements. Our evaluation of the dataset reveals that these RL-based frameworks achieve a substantial improvements in 63-66% of the tasks compared to previous approaches. 

---
# Think in Safety: Unveiling and Mitigating Safety Alignment Collapse in Multimodal Large Reasoning Model 

**Authors**: Xinyue Lou, You Li, Jinan Xu, Xiangyu Shi, Chi Chen, Kaiyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06538)  

**Abstract**: The rapid development of multimodal large reasoning models (MLRMs) has demonstrated broad application potential, yet their safety and reliability remain critical concerns that require systematic exploration. To address this gap, we conduct a comprehensive and systematic safety evaluation of 11 MLRMs across 5 benchmarks and unveil prevalent safety degradation phenomena in most advanced models. Moreover, our analysis reveals distinct safety patterns across different benchmarks: significant safety degradation is observed across jailbreak robustness benchmarks, whereas safety-awareness benchmarks demonstrate less pronounced degradation. In particular, a long thought process in some scenarios even enhances safety performance. Therefore, it is a potential approach to addressing safety issues in MLRMs by leveraging the intrinsic reasoning capabilities of the model to detect unsafe intent. To operationalize this insight, we construct a multimodal tuning dataset that incorporates a safety-oriented thought process. Experimental results from fine-tuning existing MLRMs with this dataset effectively enhances the safety on both jailbreak robustness and safety-awareness benchmarks. This study provides a new perspective for developing safe MLRMs. Our dataset is available at this https URL. 

---
# xGen-small Technical Report 

**Authors**: Erik Nijkamp, Bo Pang, Egor Pakhomov, Akash Gokul, Jin Qu, Silvio Savarese, Yingbo Zhou, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.06496)  

**Abstract**: We introduce xGen-small, a family of 4B and 9B Transformer decoder models optimized for long-context applications. Our vertically integrated pipeline unites domain-balanced, frequency-aware data curation; multi-stage pre-training with quality annealing and length extension to 128k tokens; and targeted post-training via supervised fine-tuning, preference learning, and online reinforcement learning. xGen-small delivers strong performance across various tasks, especially in math and coding domains, while excelling at long context benchmarks. 

---
# Is your multimodal large language model a good science tutor? 

**Authors**: Ming Liu, Liwen Wang, Wensheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06418)  

**Abstract**: Multimodal large language models (MLLMs) demonstrate impressive performance on scientific reasoning tasks (e.g., ScienceQA). However, most existing benchmarks focus narrowly on the accuracy of the final answer while ignoring other metrics. In particular, when applying MLLMs to educational contexts, the goal is not only correctness but also the ability to teach. In this paper, we propose a framework that evaluates MLLMs as science tutors using a comprehensive educational rubric and a simulated student model that judges the teaching performance of the tutors. Given a list of candidate MLLM science tutors, we use rubric-based student judgments to produce a range of tutor performance scores, identifying both strong and weak tutors. Using the training section of the ScienceQA dataset, we then construct a data set of pairwise comparisons between the outputs of strong and weak tutors. This enables us to apply multiple preference optimization methods to fine-tune an underperforming tutor model (Qwen2-VL-2B) into more effective ones. Our results also show that strong problem-solving skills do not guarantee high-quality tutoring and that performance optimization-guided refinements can yield more educationally aligned tutor models. This approach opens avenues for building MLLMs that serve not only as problem solvers, but as genuinely helpful educational assistants. 

---
# ScaleMCP: Dynamic and Auto-Synchronizing Model Context Protocol Tools for LLM Agents 

**Authors**: Elias Lumer, Anmol Gulati, Vamse Kumar Subbiah, Pradeep Honaganahalli Basavaraju, James A. Burke  

**Link**: [PDF](https://arxiv.org/pdf/2505.06416)  

**Abstract**: Recent advancements in Large Language Models (LLMs) and the introduction of the Model Context Protocol (MCP) have significantly expanded LLM agents' capability to interact dynamically with external tools and APIs. However, existing tool selection frameworks do not integrate MCP servers, instead relying heavily on error-prone manual updates to monolithic local tool repositories, leading to duplication, inconsistencies, and inefficiencies. Additionally, current approaches abstract tool selection before the LLM agent is invoked, limiting its autonomy and hindering dynamic re-querying capabilities during multi-turn interactions. To address these issues, we introduce ScaleMCP, a novel tool selection approach that dynamically equips LLM agents with a MCP tool retriever, giving agents the autonomy to add tools into their memory, as well as an auto-synchronizing tool storage system pipeline through CRUD (create, read, update, delete) operations with MCP servers as the single source of truth. We also propose a novel embedding strategy, Tool Document Weighted Average (TDWA), designed to selectively emphasize critical components of tool documents (e.g. tool name or synthetic questions) during the embedding process. Comprehensive evaluations conducted on a created dataset of 5,000 financial metric MCP servers, across 10 LLM models, 5 embedding models, and 5 retriever types, demonstrate substantial improvements in tool retrieval and agent invocation performance, emphasizing ScaleMCP's effectiveness in scalable, dynamic tool selection and invocation. 

---
# Enhancing Code Generation via Bidirectional Comment-Level Mutual Grounding 

**Authors**: Yifeng Di, Tianyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07768)  

**Abstract**: Large Language Models (LLMs) have demonstrated unprecedented capability in code generation. However, LLM-generated code is still plagued with a wide range of functional errors, especially for complex programming tasks that LLMs have not seen before. Recent studies have shown that developers often struggle with inspecting and fixing incorrect code generated by LLMs, diminishing their productivity and trust in LLM-based code generation. Inspired by the mutual grounding theory in communication, we propose an interactive approach that leverages code comments as a medium for developers and LLMs to establish a shared understanding. Our approach facilitates iterative grounding by interleaving code generation, inline comment generation, and contextualized user feedback through editable comments to align generated code with developer intent. We evaluated our approach on two popular benchmarks and demonstrated that our approach significantly improved multiple state-of-the-art LLMs, e.g., 17.1% pass@1 improvement for code-davinci-002 on HumanEval. Furthermore, we conducted a user study with 12 participants in comparison to two baselines: (1) interacting with GitHub Copilot, and (2) interacting with a multi-step code generation paradigm called Multi-Turn Program Synthesis. Participants completed the given programming tasks 16.7% faster and with 10.5% improvement in task success rate when using our approach. Both results show that interactively refining code comments enables the collaborative establishment of mutual grounding, leading to more accurate code generation and higher developer confidence. 

---
# Through the Looking Glass: Common Sense Consistency Evaluation of Weird Images 

**Authors**: Elisei Rykov, Kseniia Petrushina, Kseniia Titova, Anton Razzhigaev, Alexander Panchenko, Vasily Konovalov  

**Link**: [PDF](https://arxiv.org/pdf/2505.07704)  

**Abstract**: Measuring how real images look is a complex task in artificial intelligence research. For example, an image of a boy with a vacuum cleaner in a desert violates common sense. We introduce a novel method, which we call Through the Looking Glass (TLG), to assess image common sense consistency using Large Vision-Language Models (LVLMs) and Transformer-based encoder. By leveraging LVLMs to extract atomic facts from these images, we obtain a mix of accurate facts. We proceed by fine-tuning a compact attention-pooling classifier over encoded atomic facts. Our TLG has achieved a new state-of-the-art performance on the WHOOPS! and WEIRD datasets while leveraging a compact fine-tuning component. 

---
# Direct Density Ratio Optimization: A Statistically Consistent Approach to Aligning Large Language Models 

**Authors**: Rei Higuchi, Taiji Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2505.07558)  

**Abstract**: Aligning large language models (LLMs) with human preferences is crucial for safe deployment, yet existing methods assume specific preference models like Bradley-Terry model. This assumption leads to statistical inconsistency, where more data doesn't guarantee convergence to true human preferences. To address this critical gap, we introduce a novel alignment method Direct Density Ratio Optimization (DDRO). DDRO directly estimates the density ratio between preferred and unpreferred output distributions, circumventing the need for explicit human preference modeling. We theoretically prove that DDRO is statistically consistent, ensuring convergence to the true preferred distribution as the data size grows, regardless of the underlying preference structure. Experiments demonstrate that DDRO achieves superior performance compared to existing methods on many major benchmarks. DDRO unlocks the potential for truly data-driven alignment, paving the way for more reliable and human-aligned LLMs. 

---
# A Survey on Collaborative Mechanisms Between Large and Small Language Models 

**Authors**: Yi Chen, JiaHao Zhao, HaoHao Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.07460)  

**Abstract**: Large Language Models (LLMs) deliver powerful AI capabilities but face deployment challenges due to high resource costs and latency, whereas Small Language Models (SLMs) offer efficiency and deployability at the cost of reduced performance. Collaboration between LLMs and SLMs emerges as a crucial paradigm to synergistically balance these trade-offs, enabling advanced AI applications, especially on resource-constrained edge devices. This survey provides a comprehensive overview of LLM-SLM collaboration, detailing various interaction mechanisms (pipeline, routing, auxiliary, distillation, fusion), key enabling technologies, and diverse application scenarios driven by on-device needs like low latency, privacy, personalization, and offline operation. While highlighting the significant potential for creating more efficient, adaptable, and accessible AI, we also discuss persistent challenges including system overhead, inter-model consistency, robust task allocation, evaluation complexity, and security/privacy concerns. Future directions point towards more intelligent adaptive frameworks, deeper model fusion, and expansion into multimodal and embodied AI, positioning LLM-SLM collaboration as a key driver for the next generation of practical and ubiquitous artificial intelligence. 

---
# Multi-Domain Audio Question Answering Toward Acoustic Content Reasoning in The DCASE 2025 Challenge 

**Authors**: Chao-Han Huck Yang, Sreyan Ghosh, Qing Wang, Jaeyeon Kim, Hengyi Hong, Sonal Kumar, Guirui Zhong, Zhifeng Kong, S Sakshi, Vaibhavi Lokegaonkar, Oriol Nieto, Ramani Duraiswami, Dinesh Manocha, Gunhee Kim, Jun Du, Rafael Valle, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2505.07365)  

**Abstract**: We present Task 5 of the DCASE 2025 Challenge: an Audio Question Answering (AQA) benchmark spanning multiple domains of sound understanding. This task defines three QA subsets (Bioacoustics, Temporal Soundscapes, and Complex QA) to test audio-language models on interactive question-answering over diverse acoustic scenes. We describe the dataset composition (from marine mammal calls to soundscapes and complex real-world clips), the evaluation protocol (top-1 accuracy with answer-shuffling robustness), and baseline systems (Qwen2-Audio-7B, AudioFlamingo 2, Gemini-2-Flash). Preliminary results on the development set are compared, showing strong variation across models and subsets. This challenge aims to advance the audio understanding and reasoning capabilities of audio-language models toward human-level acuity, which are crucial for enabling AI agents to perceive and interact about the world effectively. 

---
# Securing Genomic Data Against Inference Attacks in Federated Learning Environments 

**Authors**: Chetan Pathade, Shubham Patil  

**Link**: [PDF](https://arxiv.org/pdf/2505.07188)  

**Abstract**: Federated Learning (FL) offers a promising framework for collaboratively training machine learning models across decentralized genomic datasets without direct data sharing. While this approach preserves data locality, it remains susceptible to sophisticated inference attacks that can compromise individual privacy. In this study, we simulate a federated learning setup using synthetic genomic data and assess its vulnerability to three key attack vectors: Membership Inference Attack (MIA), Gradient-Based Membership Inference Attack, and Label Inference Attack (LIA). Our experiments reveal that Gradient-Based MIA achieves the highest effectiveness, with a precision of 0.79 and F1-score of 0.87, underscoring the risk posed by gradient exposure in federated updates. Additionally, we visualize comparative attack performance through radar plots and quantify model leakage across clients. The findings emphasize the inadequacy of naïve FL setups in safeguarding genomic privacy and motivate the development of more robust privacy-preserving mechanisms tailored to the unique sensitivity of genomic data. 

---
# One Trigger Token Is Enough: A Defense Strategy for Balancing Safety and Usability in Large Language Models 

**Authors**: Haoran Gu, Handing Wang, Yi Mei, Mengjie Zhang, Yaochu Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07167)  

**Abstract**: Large Language Models (LLMs) have been extensively used across diverse domains, including virtual assistants, automated code generation, and scientific research. However, they remain vulnerable to jailbreak attacks, which manipulate the models into generating harmful responses despite safety alignment. Recent studies have shown that current safety-aligned LLMs often undergo the shallow safety alignment, where the first few tokens largely determine whether the response will be harmful. Through comprehensive observations, we find that safety-aligned LLMs and various defense strategies generate highly similar initial tokens in their refusal responses, which we define as safety trigger tokens. Building on this insight, we propose \texttt{D-STT}, a simple yet effective defense algorithm that identifies and explicitly decodes safety trigger tokens of the given safety-aligned LLM to trigger the model's learned safety patterns. In this process, the safety trigger is constrained to a single token, which effectively preserves model usability by introducing minimum intervention in the decoding process. Extensive experiments across diverse jailbreak attacks and benign prompts demonstrate that \ours significantly reduces output harmfulness while preserving model usability and incurring negligible response time overhead, outperforming ten baseline methods. 

---
# Pre-training vs. Fine-tuning: A Reproducibility Study on Dense Retrieval Knowledge Acquisition 

**Authors**: Zheng Yao, Shuai Wang, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2505.07166)  

**Abstract**: Dense retrievers utilize pre-trained backbone language models (e.g., BERT, LLaMA) that are fine-tuned via contrastive learning to perform the task of encoding text into sense representations that can be then compared via a shallow similarity operation, e.g. inner product. Recent research has questioned the role of fine-tuning vs. that of pre-training within dense retrievers, specifically arguing that retrieval knowledge is primarily gained during pre-training, meaning knowledge not acquired during pre-training cannot be sub-sequentially acquired via fine-tuning. We revisit this idea here as the claim was only studied in the context of a BERT-based encoder using DPR as representative dense retriever. We extend the previous analysis by testing other representation approaches (comparing the use of CLS tokens with that of mean pooling), backbone architectures (encoder-only BERT vs. decoder-only LLaMA), and additional datasets (MSMARCO in addition to Natural Questions). Our study confirms that in DPR tuning, pre-trained knowledge underpins retrieval performance, with fine-tuning primarily adjusting neuron activation rather than reorganizing knowledge. However, this pattern does not hold universally, such as in mean-pooled (Contriever) and decoder-based (LLaMA) models. We ensure full reproducibility and make our implementation publicly available at this https URL. 

---
# Reassessing Large Language Model Boolean Query Generation for Systematic Reviews 

**Authors**: Shuai Wang, Harrisen Scells, Bevan Koopman, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2505.07155)  

**Abstract**: Systematic reviews are comprehensive literature reviews that address highly focused research questions and represent the highest form of evidence in medicine. A critical step in this process is the development of complex Boolean queries to retrieve relevant literature. Given the difficulty of manually constructing these queries, recent efforts have explored Large Language Models (LLMs) to assist in their formulation. One of the first studies,Wang et al., investigated ChatGPT for this task, followed by Staudinger et al., which evaluated multiple LLMs in a reproducibility study. However, the latter overlooked several key aspects of the original work, including (i) validation of generated queries, (ii) output formatting constraints, and (iii) selection of examples for chain-of-thought (Guided) prompting. As a result, its findings diverged significantly from the original study. In this work, we systematically reproduce both studies while addressing these overlooked factors. Our results show that query effectiveness varies significantly across models and prompt designs, with guided query formulation benefiting from well-chosen seed studies. Overall, prompt design and model selection are key drivers of successful query formulation. Our findings provide a clearer understanding of LLMs' potential in Boolean query generation and highlight the importance of model- and prompt-specific optimisations. The complex nature of systematic reviews adds to challenges in both developing and reproducing methods but also highlights the importance of reproducibility studies in this domain. 

---
# LLM-Augmented Chemical Synthesis and Design Decision Programs 

**Authors**: Haorui Wang, Jeff Guo, Lingkai Kong, Rampi Ramprasad, Philippe Schwaller, Yuanqi Du, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07027)  

**Abstract**: Retrosynthesis, the process of breaking down a target molecule into simpler precursors through a series of valid reactions, stands at the core of organic chemistry and drug development. Although recent machine learning (ML) research has advanced single-step retrosynthetic modeling and subsequent route searches, these solutions remain restricted by the extensive combinatorial space of possible pathways. Concurrently, large language models (LLMs) have exhibited remarkable chemical knowledge, hinting at their potential to tackle complex decision-making tasks in chemistry. In this work, we explore whether LLMs can successfully navigate the highly constrained, multi-step retrosynthesis planning problem. We introduce an efficient scheme for encoding reaction pathways and present a new route-level search strategy, moving beyond the conventional step-by-step reactant prediction. Through comprehensive evaluations, we show that our LLM-augmented approach excels at retrosynthesis planning and extends naturally to the broader challenge of synthesizable molecular design. 

---
# Towards the Three-Phase Dynamics of Generalization Power of a DNN 

**Authors**: Yuxuan He, Junpeng Zhang, Hongyuan Zhang, Quanshi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06993)  

**Abstract**: This paper proposes a new perspective for analyzing the generalization power of deep neural networks (DNNs), i.e., directly disentangling and analyzing the dynamics of generalizable and non-generalizable interaction encoded by a DNN through the training process. Specifically, this work builds upon the recent theoretical achievement in explainble AI, which proves that the detailed inference logic of DNNs can be can be strictly rewritten as a small number of AND-OR interaction patterns. Based on this, we propose an efficient method to quantify the generalization power of each interaction, and we discover a distinct three-phase dynamics of the generalization power of interactions during training. In particular, the early phase of training typically removes noisy and non-generalizable interactions and learns simple and generalizable ones. The second and the third phases tend to capture increasingly complex interactions that are harder to generalize. Experimental results verify that the learning of non-generalizable interactions is the the direct cause for the gap between the training and testing losses. 

---
# Web Page Classification using LLMs for Crawling Support 

**Authors**: Yuichi Sasazawa, Yasuhiro Sogawa  

**Link**: [PDF](https://arxiv.org/pdf/2505.06972)  

**Abstract**: A web crawler is a system designed to collect web pages, and efficient crawling of new pages requires appropriate algorithms. While website features such as XML sitemaps and the frequency of past page updates provide important clues for accessing new pages, their universal application across diverse conditions is challenging. In this study, we propose a method to efficiently collect new pages by classifying web pages into two types, "Index Pages" and "Content Pages," using a large language model (LLM), and leveraging the classification results to select index pages as starting points for accessing new pages. We construct a dataset with automatically annotated web page types and evaluate our approach from two perspectives: the page type classification performance and coverage of new pages. Experimental results demonstrate that the LLM-based method outperformed baseline methods in both evaluation metrics. 

---
# A digital perspective on the role of a stemma in material-philological transmission studies 

**Authors**: Katarzyna Anna Kapitan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06938)  

**Abstract**: Taking its point of departure in the recent developments in the field of digital humanities and the increasing automatisation of scholarly workflows, this study explores the implications of digital approaches to textual traditions for the broader field of textual scholarship. It argues that the relative simplicity of creating computergenerated stemmas allows us to view the stemma codicum as a research tool rather than the final product of our scholarly investigation. Using the Old Norse saga of Hrómundur as a case study, this article demonstrates that stemmas can serve as a starting point for exploring textual traditions further. In doing so, they enable us to address research questions that otherwise remain unanswered. The article is accompanied by datasets used to generate stemmas for the Hrómundar saga tradition as well as two custom Python scripts. The scripts are designed to convert XML-based textual data, encoded according to the TEI Guidelines, into the input format used for the analysis in the PHYLIP package to generate unrooted trees of relationships between texts. 

---
# Multi-Modal Explainable Medical AI Assistant for Trustworthy Human-AI Collaboration 

**Authors**: Honglong Yang, Shanshan Song, Yi Qin, Lehan Wang, Haonan Wang, Xinpeng Ding, Qixiang Zhang, Bodong Du, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.06898)  

**Abstract**: Generalist Medical AI (GMAI) systems have demonstrated expert-level performance in biomedical perception tasks, yet their clinical utility remains limited by inadequate multi-modal explainability and suboptimal prognostic capabilities. Here, we present XMedGPT, a clinician-centric, multi-modal AI assistant that integrates textual and visual interpretability to support transparent and trustworthy medical decision-making. XMedGPT not only produces accurate diagnostic and descriptive outputs, but also grounds referenced anatomical sites within medical images, bridging critical gaps in interpretability and enhancing clinician usability. To support real-world deployment, we introduce a reliability indexing mechanism that quantifies uncertainty through consistency-based assessment via interactive question-answering. We validate XMedGPT across four pillars: multi-modal interpretability, uncertainty quantification, and prognostic modeling, and rigorous benchmarking. The model achieves an IoU of 0.703 across 141 anatomical regions, and a Kendall's tau-b of 0.479, demonstrating strong alignment between visual rationales and clinical outcomes. For uncertainty estimation, it attains an AUC of 0.862 on visual question answering and 0.764 on radiology report generation. In survival and recurrence prediction for lung and glioma cancers, it surpasses prior leading models by 26.9%, and outperforms GPT-4o by 25.0%. Rigorous benchmarking across 347 datasets covers 40 imaging modalities and external validation spans 4 anatomical systems confirming exceptional generalizability, with performance gains surpassing existing GMAI by 20.7% for in-domain evaluation and 16.7% on 11,530 in-house data evaluation. Together, XMedGPT represents a significant leap forward in clinician-centric AI integration, offering trustworthy and scalable support for diverse healthcare applications. 

---
# Benign Samples Matter! Fine-tuning On Outlier Benign Samples Severely Breaks Safety 

**Authors**: Zihan Guan, Mengxuan Hu, Ronghang Zhu, Sheng Li, Anil Vullikanti  

**Link**: [PDF](https://arxiv.org/pdf/2505.06843)  

**Abstract**: Recent studies have uncovered a troubling vulnerability in the fine-tuning stage of large language models (LLMs): even fine-tuning on entirely benign datasets can lead to a significant increase in the harmfulness of LLM outputs. Building on this finding, our red teaming study takes this threat one step further by developing a more effective attack. Specifically, we analyze and identify samples within benign datasets that contribute most to safety degradation, then fine-tune LLMs exclusively on these samples. We approach this problem from an outlier detection perspective and propose Self-Inf-N, to detect and extract outliers for fine-tuning. Our findings reveal that fine-tuning LLMs on 100 outlier samples selected by Self-Inf-N in the benign datasets severely compromises LLM safety alignment. Extensive experiments across seven mainstream LLMs demonstrate that our attack exhibits high transferability across different architectures and remains effective in practical scenarios. Alarmingly, our results indicate that most existing mitigation strategies fail to defend against this attack, underscoring the urgent need for more robust alignment safeguards. Codes are available at this https URL. 

---
# Overview of the NLPCC 2025 Shared Task 4: Multi-modal, Multilingual, and Multi-hop Medical Instructional Video Question Answering Challenge 

**Authors**: Bin Li, Shenxi Liu, Yixuan Weng, Yue Du, Yuhang Tian, Shoujun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.06814)  

**Abstract**: Following the successful hosts of the 1-st (NLPCC 2023 Foshan) CMIVQA and the 2-rd (NLPCC 2024 Hangzhou) MMIVQA challenges, this year, a new task has been introduced to further advance research in multi-modal, multilingual, and multi-hop medical instructional question answering (M4IVQA) systems, with a specific focus on medical instructional videos. The M4IVQA challenge focuses on evaluating models that integrate information from medical instructional videos, understand multiple languages, and answer multi-hop questions requiring reasoning over various modalities. This task consists of three tracks: multi-modal, multilingual, and multi-hop Temporal Answer Grounding in Single Video (M4TAGSV), multi-modal, multilingual, and multi-hop Video Corpus Retrieval (M4VCR) and multi-modal, multilingual, and multi-hop Temporal Answer Grounding in Video Corpus (M4TAGVC). Participants in M4IVQA are expected to develop algorithms capable of processing both video and text data, understanding multilingual queries, and providing relevant answers to multi-hop medical questions. We believe the newly introduced M4IVQA challenge will drive innovations in multimodal reasoning systems for healthcare scenarios, ultimately contributing to smarter emergency response systems and more effective medical education platforms in multilingual communities. Our official website is this https URL 

---
# Bridging Ears and Eyes: Analyzing Audio and Visual Large Language Models to Humans in Visible Sound Recognition and Reducing Their Sensory Gap via Cross-Modal Distillation 

**Authors**: Xilin Jiang, Junkai Wu, Vishal Choudhari, Nima Mesgarani  

**Link**: [PDF](https://arxiv.org/pdf/2505.06803)  

**Abstract**: Audio large language models (LLMs) are considered experts at recognizing sound objects, yet their performance relative to LLMs in other sensory modalities, such as visual or audio-visual LLMs, and to humans using their ears, eyes, or both remains unexplored. To investigate this, we systematically evaluate audio, visual, and audio-visual LLMs, specifically Qwen2-Audio, Qwen2-VL, and Qwen2.5-Omni, against humans in recognizing sound objects of different classes from audio-only, silent video, or sounded video inputs. We uncover a performance gap between Qwen2-Audio and Qwen2-VL that parallels the sensory discrepancy between human ears and eyes. To reduce this gap, we introduce a cross-modal distillation framework, where an LLM in one modality serves as the teacher and another as the student, with knowledge transfer in sound classes predicted as more challenging to the student by a heuristic model. Distillation in both directions, from Qwen2-VL to Qwen2-Audio and vice versa, leads to notable improvements, particularly in challenging classes. This work highlights the sensory gap in LLMs from a human-aligned perspective and proposes a principled approach to enhancing modality-specific perception in multimodal LLMs. 

---
# Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4): Analysis and Variations 

**Authors**: Patrick Blumenberg, Thomas Graave, Tim Fingscheidt  

**Link**: [PDF](https://arxiv.org/pdf/2505.06653)  

**Abstract**: Large language models (LLMs) demand extensive memory capacity during both fine-tuning and inference. To enable memory-efficient fine-tuning, existing methods apply block-wise quantization techniques, such as NF4 and AF4, to the network weights. We show that these quantization techniques incur suboptimal quantization errors. Therefore, as a first novelty, we propose an optimization approach for block-wise quantization. Using this method, we design a family of quantizers named 4-bit block-wise optimal float (BOF4), which consistently reduces the quantization error compared to both baseline methods. We provide both a theoretical and a data-driven solution for the optimization process and prove their practical equivalence. Secondly, we propose a modification to the employed normalization method based on the signed absolute block maximum (BOF4-S), enabling further reduction of the quantization error and empirically achieving less degradation in language modeling performance. Thirdly, we explore additional variations of block-wise quantization methods applied to LLMs through an experimental study on the importance of accurately representing zero and large-amplitude weights on the one hand, and optimization towards various error metrics on the other hand. Lastly, we introduce a mixed-precision quantization strategy dubbed outlier-preserving quantization (OPQ) to address the distributional mismatch induced by outlier weights in block-wise quantization. By storing outlier weights in 16-bit precision (OPQ) while applying BOF4-S, we achieve top performance among 4-bit block-wise quantization techniques w.r.t. perplexity. 

---
# Divide (Text) and Conquer (Sentiment): Improved Sentiment Classification by Constituent Conflict Resolution 

**Authors**: Jan Kościałkowski, Paweł Marcinkowski  

**Link**: [PDF](https://arxiv.org/pdf/2505.06320)  

**Abstract**: Sentiment classification, a complex task in natural language processing, becomes even more challenging when analyzing passages with multiple conflicting tones. Typically, longer passages exacerbate this issue, leading to decreased model performance. The aim of this paper is to introduce novel methodologies for isolating conflicting sentiments and aggregating them to effectively predict the overall sentiment of such passages. One of the aggregation strategies involves a Multi-Layer Perceptron (MLP) model which outperforms baseline models across various datasets, including Amazon, Twitter, and SST while costing $\sim$1/100 of what fine-tuning the baseline would take. 

---
# AI Approaches to Qualitative and Quantitative News Analytics on NATO Unity 

**Authors**: Bohdan M. Pavlyshenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.06313)  

**Abstract**: The paper considers the use of GPT models with retrieval-augmented generation (RAG) for qualitative and quantitative analytics on NATO sentiments, NATO unity and NATO Article 5 trust opinion scores in different web sources: news sites found via Google Search API, Youtube videos with comments, and Reddit discussions. A RAG approach using GPT-4.1 model was applied to analyse news where NATO related topics were discussed. Two levels of RAG analytics were used: on the first level, the GPT model generates qualitative news summaries and quantitative opinion scores using zero-shot prompts; on the second level, the GPT model generates the summary of news summaries. Quantitative news opinion scores generated by the GPT model were analysed using Bayesian regression to get trend lines. The distributions found for the regression parameters make it possible to analyse an uncertainty in specified news opinion score trends. Obtained results show a downward trend for analysed scores of opinion related to NATO unity.
This approach does not aim to conduct real political analysis; rather, it consider AI based approaches which can be used for further analytics
as a part of a complex analytical approach. The obtained results demonstrate that the use of GPT models for news analysis can give informative qualitative and quantitative analytics, providing important insights.
The dynamic model based on neural ordinary differential equations was considered for modelling public opinions. This approach makes it possible to analyse different scenarios for evolving public opinions. 

---
# Lossless Compression of Large Language Model-Generated Text via Next-Token Prediction 

**Authors**: Yu Mao, Holger Pirk, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2505.06297)  

**Abstract**: As large language models (LLMs) continue to be deployed and utilized across domains, the volume of LLM-generated data is growing rapidly. This trend highlights the increasing importance of effective and lossless compression for such data in modern text management systems. However, compressing LLM-generated data presents unique challenges compared to traditional human- or machine-generated content. Traditional machine-generated data is typically derived from computational processes or device outputs, often highly structured and limited to low-level elements like labels or numerical values. This structure enables conventional lossless compressors to perform efficiently. In contrast, LLM-generated data is more complex and diverse, requiring new approaches for effective compression. In this work, we conduct the first systematic investigation of lossless compression techniques tailored specifically to LLM-generated data. Notably, because LLMs are trained via next-token prediction, we find that LLM-generated data is highly predictable for the models themselves. This predictability enables LLMs to serve as efficient compressors of their own outputs. Through extensive experiments with 14 representative LLMs and 8 LLM-generated datasets from diverse domains, we show that LLM-based prediction methods achieve remarkable compression rates, exceeding 20x, far surpassing the 3x rate achieved by Gzip, a widely used general-purpose compressor. Furthermore, this advantage holds across different LLM sizes and dataset types, demonstrating the robustness and practicality of LLM-based methods in lossless text compression under generative AI workloads. 

---
