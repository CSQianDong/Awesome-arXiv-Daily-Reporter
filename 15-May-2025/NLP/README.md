# WorldView-Bench: A Benchmark for Evaluating Global Cultural Perspectives in Large Language Models 

**Authors**: Abdullah Mushtaq, Imran Taj, Rafay Naeem, Ibrahim Ghaznavi, Junaid Qadir  

**Link**: [PDF](https://arxiv.org/pdf/2505.09595)  

**Abstract**: Large Language Models (LLMs) are predominantly trained and aligned in ways that reinforce Western-centric epistemologies and socio-cultural norms, leading to cultural homogenization and limiting their ability to reflect global civilizational plurality. Existing benchmarking frameworks fail to adequately capture this bias, as they rely on rigid, closed-form assessments that overlook the complexity of cultural inclusivity. To address this, we introduce WorldView-Bench, a benchmark designed to evaluate Global Cultural Inclusivity (GCI) in LLMs by analyzing their ability to accommodate diverse worldviews. Our approach is grounded in the Multiplex Worldview proposed by Senturk et al., which distinguishes between Uniplex models, reinforcing cultural homogenization, and Multiplex models, which integrate diverse perspectives. WorldView-Bench measures Cultural Polarization, the exclusion of alternative perspectives, through free-form generative evaluation rather than conventional categorical benchmarks. We implement applied multiplexity through two intervention strategies: (1) Contextually-Implemented Multiplex LLMs, where system prompts embed multiplexity principles, and (2) Multi-Agent System (MAS)-Implemented Multiplex LLMs, where multiple LLM agents representing distinct cultural perspectives collaboratively generate responses. Our results demonstrate a significant increase in Perspectives Distribution Score (PDS) entropy from 13% at baseline to 94% with MAS-Implemented Multiplex LLMs, alongside a shift toward positive sentiment (67.7%) and enhanced cultural balance. These findings highlight the potential of multiplex-aware AI evaluation in mitigating cultural bias in LLMs, paving the way for more inclusive and ethically aligned AI systems. 

---
# PT-MoE: An Efficient Finetuning Framework for Integrating Mixture-of-Experts into Prompt Tuning 

**Authors**: Zongqian Li, Yixuan Su, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2505.09519)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) methods have shown promise in adapting large language models, yet existing approaches exhibit counter-intuitive phenomena: integrating router into prompt tuning (PT) increases training efficiency yet does not improve performance universally; parameter reduction through matrix decomposition can improve performance in specific domains. Motivated by these observations and the modular nature of PT, we propose PT-MoE, a novel framework that integrates matrix decomposition with mixture-of-experts (MoE) routing for efficient PT. Results across 17 datasets demonstrate that PT-MoE achieves state-of-the-art performance in both question answering (QA) and mathematical problem solving tasks, improving F1 score by 1.49 points over PT and 2.13 points over LoRA in QA tasks, while enhancing mathematical accuracy by 10.75 points over PT and 0.44 points over LoRA, all while using 25% fewer parameters than LoRA. Our analysis reveals that while PT methods generally excel in QA tasks and LoRA-based methods in math datasets, the integration of matrix decomposition and MoE in PT-MoE yields complementary benefits: decomposition enables efficient parameter sharing across experts while MoE provides dynamic adaptation, collectively enabling PT-MoE to demonstrate cross-task consistency and generalization abilities. These findings, along with ablation studies on routing mechanisms and architectural components, provide insights for future PEFT methods. 

---
# Multilingual Machine Translation with Quantum Encoder Decoder Attention-based Convolutional Variational Circuits 

**Authors**: Subrit Dikshit, Ritu Tiwari, Priyank Jain  

**Link**: [PDF](https://arxiv.org/pdf/2505.09407)  

**Abstract**: Cloud-based multilingual translation services like Google Translate and Microsoft Translator achieve state-of-the-art translation capabilities. These services inherently use large multilingual language models such as GRU, LSTM, BERT, GPT, T5, or similar encoder-decoder architectures with attention mechanisms as the backbone. Also, new age natural language systems, for instance ChatGPT and DeepSeek, have established huge potential in multiple tasks in natural language processing. At the same time, they also possess outstanding multilingual translation capabilities. However, these models use the classical computing realm as a backend. QEDACVC (Quantum Encoder Decoder Attention-based Convolutional Variational Circuits) is an alternate solution that explores the quantum computing realm instead of the classical computing realm to study and demonstrate multilingual machine translation. QEDACVC introduces the quantum encoder-decoder architecture that simulates and runs on quantum computing hardware via quantum convolution, quantum pooling, quantum variational circuit, and quantum attention as software alterations. QEDACVC achieves an Accuracy of 82% when trained on the OPUS dataset for English, French, German, and Hindi corpora for multilingual translations. 

---
# Qwen3 Technical Report 

**Authors**: An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, Zihan Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09388)  

**Abstract**: In this work, we present Qwen3, the latest version of the Qwen model family. Qwen3 comprises a series of large language models (LLMs) designed to advance performance, efficiency, and multilingual capabilities. The Qwen3 series includes models of both dense and Mixture-of-Expert (MoE) architectures, with parameter scales ranging from 0.6 to 235 billion. A key innovation in Qwen3 is the integration of thinking mode (for complex, multi-step reasoning) and non-thinking mode (for rapid, context-driven responses) into a unified framework. This eliminates the need to switch between different models--such as chat-optimized models (e.g., GPT-4o) and dedicated reasoning models (e.g., QwQ-32B)--and enables dynamic mode switching based on user queries or chat templates. Meanwhile, Qwen3 introduces a thinking budget mechanism, allowing users to allocate computational resources adaptively during inference, thereby balancing latency and performance based on task complexity. Moreover, by leveraging the knowledge from the flagship models, we significantly reduce the computational resources required to build smaller-scale models, while ensuring their highly competitive performance. Empirical evaluations demonstrate that Qwen3 achieves state-of-the-art results across diverse benchmarks, including tasks in code generation, mathematical reasoning, agent tasks, etc., competitive against larger MoE models and proprietary models. Compared to its predecessor Qwen2.5, Qwen3 expands multilingual support from 29 to 119 languages and dialects, enhancing global accessibility through improved cross-lingual understanding and generation capabilities. To facilitate reproducibility and community-driven research and development, all Qwen3 models are publicly accessible under Apache 2.0. 

---
# Llama See, Llama Do: A Mechanistic Perspective on Contextual Entrainment and Distraction in LLMs 

**Authors**: Jingcheng Niu, Xingdi Yuan, Tong Wang, Hamidreza Saghir, Amir H. Abdi  

**Link**: [PDF](https://arxiv.org/pdf/2505.09338)  

**Abstract**: We observe a novel phenomenon, contextual entrainment, across a wide range of language models (LMs) and prompt settings, providing a new mechanistic perspective on how LMs become distracted by ``irrelevant'' contextual information in the input prompt. Specifically, LMs assign significantly higher logits (or probabilities) to any tokens that have previously appeared in the context prompt, even for random tokens. This suggests that contextual entrainment is a mechanistic phenomenon, occurring independently of the relevance or semantic relation of the tokens to the question or the rest of the sentence. We find statistically significant evidence that the magnitude of contextual entrainment is influenced by semantic factors. Counterfactual prompts have a greater effect compared to factual ones, suggesting that while contextual entrainment is a mechanistic phenomenon, it is modulated by semantic factors.
We hypothesise that there is a circuit of attention heads -- the entrainment heads -- that corresponds to the contextual entrainment phenomenon. Using a novel entrainment head discovery method based on differentiable masking, we identify these heads across various settings. When we ``turn off'' these heads, i.e., set their outputs to zero, the effect of contextual entrainment is significantly attenuated, causing the model to generate output that capitulates to what it would produce if no distracting context were provided. Our discovery of contextual entrainment, along with our investigation into LM distraction via the entrainment heads, marks a key step towards the mechanistic analysis and mitigation of the distraction problem. 

---
# Scent of Knowledge: Optimizing Search-Enhanced Reasoning with Information Foraging 

**Authors**: Hongjin Qian, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09316)  

**Abstract**: Augmenting large language models (LLMs) with external retrieval has become a standard method to address their inherent knowledge cutoff limitations. However, traditional retrieval-augmented generation methods employ static, pre-inference retrieval strategies, making them inadequate for complex tasks involving ambiguous, multi-step, or evolving information needs. Recent advances in test-time scaling techniques have demonstrated significant potential in enabling LLMs to dynamically interact with external tools, motivating the shift toward adaptive inference-time retrieval. Inspired by Information Foraging Theory (IFT), we propose InForage, a reinforcement learning framework that formalizes retrieval-augmented reasoning as a dynamic information-seeking process. Unlike existing approaches, InForage explicitly rewards intermediate retrieval quality, encouraging LLMs to iteratively gather and integrate information through adaptive search behaviors. To facilitate training, we construct a human-guided dataset capturing iterative search and reasoning trajectories for complex, real-world web tasks. Extensive evaluations across general question answering, multi-hop reasoning tasks, and a newly developed real-time web QA dataset demonstrate InForage's superior performance over baseline methods. These results highlight InForage's effectiveness in building robust, adaptive, and efficient reasoning agents. 

---
# A Scalable Unsupervised Framework for multi-aspect labeling of Multilingual and Multi-Domain Review Data 

**Authors**: Jiin Park, Misuk Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.09286)  

**Abstract**: Effectively analyzing online review data is essential across industries. However, many existing studies are limited to specific domains and languages or depend on supervised learning approaches that require large-scale labeled datasets. To address these limitations, we propose a multilingual, scalable, and unsupervised framework for cross-domain aspect detection. This framework is designed for multi-aspect labeling of multilingual and multi-domain review data. In this study, we apply automatic labeling to Korean and English review datasets spanning various domains and assess the quality of the generated labels through extensive experiments. Aspect category candidates are first extracted through clustering, and each review is then represented as an aspect-aware embedding vector using negative sampling. To evaluate the framework, we conduct multi-aspect labeling and fine-tune several pretrained language models to measure the effectiveness of the automatically generated labels. Results show that these models achieve high performance, demonstrating that the labels are suitable for training. Furthermore, comparisons with publicly available large language models highlight the framework's superior consistency and scalability when processing large-scale data. A human evaluation also confirms that the quality of the automatic labels is comparable to those created manually. This study demonstrates the potential of a robust multi-aspect labeling approach that overcomes limitations of supervised methods and is adaptable to multilingual, multi-domain environments. Future research will explore automatic review summarization and the integration of artificial intelligence agents to further improve the efficiency and depth of review analysis. 

---
# How an unintended Side Effect of a Research Project led to Boosting the Power of UML 

**Authors**: Ulrich Frank, Pierre Maier  

**Link**: [PDF](https://arxiv.org/pdf/2505.09269)  

**Abstract**: This paper describes the design, implementation and use of a new UML modeling tool that represents a significant advance over conventional tools. Among other things, it allows the integration of class diagrams and object diagrams as well as the execution of objects. This not only enables new software architectures characterized by the integration of software with corresponding object models, but is also ideal for use in teaching, as it provides students with a particularly stimulating learning experience. A special feature of the project is that it has emerged from a long-standing international research project, which is aimed at a comprehensive multi-level architecture. The project is therefore an example of how research can lead to valuable results that arise as a side effect of other work. 

---
# CEC-Zero: Chinese Error Correction Solution Based on LLM 

**Authors**: Sophie Zhang, Zhiming Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.09082)  

**Abstract**: Recent advancements in large language models (LLMs) demonstrate exceptional Chinese text processing capabilities, particularly in Chinese Spelling Correction (CSC). While LLMs outperform traditional BERT-based models in accuracy and robustness, challenges persist in reliability and generalization. This paper proposes CEC-Zero, a novel reinforcement learning (RL) framework enabling LLMs to self-correct through autonomous error strategy learning without external supervision. By integrating RL with LLMs' generative power, the method eliminates dependency on annotated data or auxiliary models. Experiments reveal RL-enhanced LLMs achieve industry-viable accuracy and superior cross-domain generalization, offering a scalable solution for reliability optimization in Chinese NLP applications. This breakthrough facilitates LLM deployment in practical Chinese text correction scenarios while establishing a new paradigm for self-improving language models. 

---
# S-DAT: A Multilingual, GenAI-Driven Framework for Automated Divergent Thinking Assessment 

**Authors**: Jennifer Haase, Paul H. P. Hanel, Sebastian Pokutta  

**Link**: [PDF](https://arxiv.org/pdf/2505.09068)  

**Abstract**: This paper introduces S-DAT (Synthetic-Divergent Association Task), a scalable, multilingual framework for automated assessment of divergent thinking (DT) -a core component of human creativity. Traditional creativity assessments are often labor-intensive, language-specific, and reliant on subjective human ratings, limiting their scalability and cross-cultural applicability. In contrast, S-DAT leverages large language models and advanced multilingual embeddings to compute semantic distance -- a language-agnostic proxy for DT. We evaluate S-DAT across eleven diverse languages, including English, Spanish, German, Russian, Hindi, and Japanese (Kanji, Hiragana, Katakana), demonstrating robust and consistent scoring across linguistic contexts. Unlike prior DAT approaches, the S-DAT shows convergent validity with other DT measures and correct discriminant validity with convergent thinking. This cross-linguistic flexibility allows for more inclusive, global-scale creativity research, addressing key limitations of earlier approaches. S-DAT provides a powerful tool for fairer, more comprehensive evaluation of cognitive flexibility in diverse populations and can be freely assessed online: this https URL. 

---
# A Comprehensive Analysis of Large Language Model Outputs: Similarity, Diversity, and Bias 

**Authors**: Brandon Smith, Mohamed Reda Bouadjenek, Tahsin Alamgir Kheya, Phillip Dawson, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2505.09056)  

**Abstract**: Large Language Models (LLMs) represent a major step toward artificial general intelligence, significantly advancing our ability to interact with technology. While LLMs perform well on Natural Language Processing tasks -- such as translation, generation, code writing, and summarization -- questions remain about their output similarity, variability, and ethical implications. For instance, how similar are texts generated by the same model? How does this compare across different models? And which models best uphold ethical standards? To investigate, we used 5{,}000 prompts spanning diverse tasks like generation, explanation, and rewriting. This resulted in approximately 3 million texts from 12 LLMs, including proprietary and open-source systems from OpenAI, Google, Microsoft, Meta, and Mistral. Key findings include: (1) outputs from the same LLM are more similar to each other than to human-written texts; (2) models like WizardLM-2-8x22b generate highly similar outputs, while GPT-4 produces more varied responses; (3) LLM writing styles differ significantly, with Llama 3 and Mistral showing higher similarity, and GPT-4 standing out for distinctiveness; (4) differences in vocabulary and tone underscore the linguistic uniqueness of LLM-generated content; (5) some LLMs demonstrate greater gender balance and reduced bias. These results offer new insights into the behavior and diversity of LLM outputs, helping guide future development and ethical evaluation. 

---
# Atomic Consistency Preference Optimization for Long-Form Question Answering 

**Authors**: Jingfeng Chen, Raghuveer Thirukovalluru, Junlin Wang, Kaiwei Luo, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2505.09039)  

**Abstract**: Large Language Models (LLMs) frequently produce factoid hallucinations - plausible yet incorrect answers. A common mitigation strategy is model alignment, which improves factual accuracy by training on curated factual and non-factual pairs. However, this approach often relies on a stronger model (e.g., GPT-4) or an external knowledge base to assess factual correctness, which may not always be accessible. To address this, we propose Atomic Consistency Preference Optimization (ACPO), a self-supervised preference-tuning method that enhances factual accuracy without external supervision. ACPO leverages atomic consistency signals, i.e., the agreement of individual facts across multiple stochastic responses, to identify high- and low-quality data pairs for model alignment. By eliminating the need for costly GPT calls, ACPO provides a scalable and efficient approach to improving factoid question-answering. Despite being self-supervised, empirical results demonstrate that ACPO outperforms FactAlign, a strong supervised alignment baseline, by 1.95 points on the LongFact and BioGen datasets, highlighting its effectiveness in enhancing factual reliability without relying on external models or knowledge bases. 

---
# For GPT-4 as with Humans: Information Structure Predicts Acceptability of Long-Distance Dependencies 

**Authors**: Nicole Cuneo, Eleanor Graves, Supantho Rakshit, Adele E. Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.09005)  

**Abstract**: It remains debated how well any LM understands natural language or generates reliable metalinguistic judgments. Moreover, relatively little work has demonstrated that LMs can represent and respect subtle relationships between form and function proposed by linguists. We here focus on a particular such relationship established in recent work: English speakers' judgments about the information structure of canonical sentences predicts independently collected acceptability ratings on corresponding 'long distance dependency' [LDD] constructions, across a wide array of base constructions and multiple types of LDDs. To determine whether any LM captures this relationship, we probe GPT-4 on the same tasks used with humans and new this http URL reveal reliable metalinguistic skill on the information structure and acceptability tasks, replicating a striking interaction between the two, despite the zero-shot, explicit nature of the tasks, and little to no chance of contamination [Studies 1a, 1b]. Study 2 manipulates the information structure of base sentences and confirms a causal relationship: increasing the prominence of a constituent in a context sentence increases the subsequent acceptability ratings on an LDD construction. The findings suggest a tight relationship between natural and GPT-4 generated English, and between information structure and syntax, which begs for further exploration. 

---
# A suite of LMs comprehend puzzle statements as well as humans 

**Authors**: Adele E Goldberg, Supantho Rakshit, Jennifer Hu, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2505.08996)  

**Abstract**: Recent claims suggest that large language models (LMs) underperform humans in comprehending minimally complex English statements (Dentella et al., 2024). Here, we revisit those findings and argue that human performance was overestimated, while LLM abilities were underestimated. Using the same stimuli, we report a preregistered study comparing human responses in two conditions: one allowed rereading (replicating the original study), and one that restricted rereading (a more naturalistic comprehension test). Human accuracy dropped significantly when rereading was restricted (73%), falling below that of Falcon-180B-Chat (76%) and GPT-4 (81%). The newer GPT-o1 model achieves perfect accuracy. Results further show that both humans and models are disproportionately challenged by queries involving potentially reciprocal actions (e.g., kissing), suggesting shared pragmatic sensitivities rather than model-specific deficits. Additional analyses using Llama-2-70B log probabilities, a recoding of open-ended model responses, and grammaticality ratings of other sentences reveal systematic underestimation of model performance. We find that GPT-4o can align with either naive or expert grammaticality judgments, depending on prompt framing. These findings underscore the need for more careful experimental design and coding practices in LLM evaluation, and they challenge the assumption that current models are inherently weaker than humans at language comprehension. 

---
# Clicking some of the silly options: Exploring Player Motivation in Static and Dynamic Educational Interactive Narratives 

**Authors**: Daeun Hwang, Samuel Shields, Alex Calderwood, Shi Johnson-Bey, Michael Mateas, Noah Wardrip-Fruin, Edward F. Melcer  

**Link**: [PDF](https://arxiv.org/pdf/2505.08891)  

**Abstract**: Motivation is an important factor underlying successful learning. Previous research has demonstrated the positive effects that static interactive narrative games can have on motivation. Concurrently, advances in AI have made dynamic and adaptive approaches to interactive narrative increasingly accessible. However, limited work has explored the impact that dynamic narratives can have on learner motivation. In this paper, we compare two versions of Academical, a choice-based educational interactive narrative game about research ethics. One version employs a traditional hand-authored branching plot (i.e., static narrative) while the other dynamically sequences plots during play (i.e., dynamic narrative). Results highlight the importance of responsive content and a variety of choices for player engagement, while also illustrating the challenge of balancing pedagogical goals with the dynamic aspects of narrative. We also discuss design implications that arise from these findings. Ultimately, this work provides initial steps to illuminate the emerging potential of AI-driven dynamic narrative in educational games. 

---
# Human-AI Collaboration or Academic Misconduct? Measuring AI Use in Student Writing Through Stylometric Evidence 

**Authors**: Eduardo Araujo Oliveira, Madhavi Mohoni, Sonsoles López-Pernas, Mohammed Saqr  

**Link**: [PDF](https://arxiv.org/pdf/2505.08828)  

**Abstract**: As human-AI collaboration becomes increasingly prevalent in educational contexts, understanding and measuring the extent and nature of such interactions pose significant challenges. This research investigates the use of authorship verification (AV) techniques not as a punitive measure, but as a means to quantify AI assistance in academic writing, with a focus on promoting transparency, interpretability, and student development. Building on prior work, we structured our investigation into three stages: dataset selection and expansion, AV method development, and systematic evaluation. Using three datasets - including a public dataset (PAN-14) and two from University of Melbourne students from various courses - we expanded the data to include LLM-generated texts, totalling 1,889 documents and 540 authorship problems from 506 students. We developed an adapted Feature Vector Difference AV methodology to construct robust academic writing profiles for students, designed to capture meaningful, individual characteristics of their writing. The method's effectiveness was evaluated across multiple scenarios, including distinguishing between student-authored and LLM-generated texts and testing resilience against LLMs' attempts to mimic student writing styles. Results demonstrate the enhanced AV classifier's ability to identify stylometric discrepancies and measure human-AI collaboration at word and sentence levels while providing educators with a transparent tool to support academic integrity investigations. This work advances AV technology, offering actionable insights into the dynamics of academic writing in an AI-driven era. 

---
# Language Agents Mirror Human Causal Reasoning Biases. How Can We Help Them Think Like Scientists? 

**Authors**: Anthony GX-Chen, Dongyan Lin, Mandana Samiei, Doina Precup, Blake A. Richards, Rob Fergus, Kenneth Marino  

**Link**: [PDF](https://arxiv.org/pdf/2505.09614)  

**Abstract**: Language model (LM) agents are increasingly used as autonomous decision-makers who need to actively gather information to guide their decisions. A crucial cognitive skill for such agents is the efficient exploration and understanding of the causal structure of the world -- key to robust, scientifically grounded reasoning. Yet, it remains unclear whether LMs possess this capability or exhibit systematic biases leading to erroneous conclusions. In this work, we examine LMs' ability to explore and infer causal relationships, using the well-established "Blicket Test" paradigm from developmental psychology. We find that LMs reliably infer the common, intuitive disjunctive causal relationships but systematically struggle with the unusual, yet equally (or sometimes even more) evidenced conjunctive ones. This "disjunctive bias" persists across model families, sizes, and prompting strategies, and performance further declines as task complexity increases. Interestingly, an analogous bias appears in human adults, suggesting that LMs may have inherited deep-seated reasoning heuristics from their training data. To this end, we quantify similarities between LMs and humans, finding that LMs exhibit adult-like inference profiles (but not children-like). Finally, we propose a test-time sampling method which explicitly samples and eliminates hypotheses about causal relationships from the LM. This scalable approach significantly reduces the disjunctive bias and moves LMs closer to the goal of scientific, causally rigorous reasoning. 

---
# Customizing a Large Language Model for VHDL Design of High-Performance Microprocessors 

**Authors**: Nicolas Dupuis, Ravi Nair, Shyam Ramji, Sean McClintock, Nishant Chauhan, Priyanka Nagpal, Bart Blaner, Ken Valk, Leon Stok, Ruchir Puri  

**Link**: [PDF](https://arxiv.org/pdf/2505.09610)  

**Abstract**: The use of Large Language Models (LLMs) in hardware design has taken off in recent years, principally through its incorporation in tools that increase chip designer productivity. There has been considerable discussion about the use of LLMs in RTL specifications of chip designs, for which the two most popular languages are Verilog and VHDL. LLMs and their use in Verilog design has received significant attention due to the higher popularity of the language, but little attention so far has been given to VHDL despite its continued popularity in the industry. There has also been little discussion about the unique needs of organizations that engage in high-performance processor design, and techniques to deploy AI solutions in these settings. In this paper, we describe our journey in developing a Large Language Model (LLM) specifically for the purpose of explaining VHDL code, a task that has particular importance in an organization with decades of experience and assets in high-performance processor design. We show how we developed test sets specific to our needs and used them for evaluating models as we performed extended pretraining (EPT) of a base LLM. Expert evaluation of the code explanations produced by the EPT model increased to 69% compared to a base model rating of 43%. We further show how we developed an LLM-as-a-judge to gauge models similar to expert evaluators. This led us to deriving and evaluating a host of new models, including an instruction-tuned version of the EPT model with an expected expert evaluator rating of 71%. Our experiments also indicate that with the potential use of newer base models, this rating can be pushed to 85% and beyond. We conclude with a discussion on further improving the quality of hardware design LLMs using exciting new developments in the Generative AI world. 

---
# CXMArena: Unified Dataset to benchmark performance in realistic CXM Scenarios 

**Authors**: Raghav Garg, Kapil Sharma, Karan Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.09436)  

**Abstract**: Large Language Models (LLMs) hold immense potential for revolutionizing Customer Experience Management (CXM), particularly in contact center operations. However, evaluating their practical utility in complex operational environments is hindered by data scarcity (due to privacy concerns) and the limitations of current benchmarks. Existing benchmarks often lack realism, failing to incorporate deep knowledge base (KB) integration, real-world noise, or critical operational tasks beyond conversational fluency. To bridge this gap, we introduce CXMArena, a novel, large-scale synthetic benchmark dataset specifically designed for evaluating AI in operational CXM contexts. Given the diversity in possible contact center features, we have developed a scalable LLM-powered pipeline that simulates the brand's CXM entities that form the foundation of our datasets-such as knowledge articles including product specifications, issue taxonomies, and contact center conversations. The entities closely represent real-world distribution because of controlled noise injection (informed by domain experts) and rigorous automated validation. Building on this, we release CXMArena, which provides dedicated benchmarks targeting five important operational tasks: Knowledge Base Refinement, Intent Prediction, Agent Quality Adherence, Article Search, and Multi-turn RAG with Integrated Tools. Our baseline experiments underscore the benchmark's difficulty: even state of the art embedding and generation models achieve only 68% accuracy on article search, while standard embedding methods yield a low F1 score of 0.3 for knowledge base refinement, highlighting significant challenges for current models necessitating complex pipelines and solutions over conventional techniques. 

---
# Focus, Merge, Rank: Improved Question Answering Based on Semi-structured Knowledge Bases 

**Authors**: Derian Boer, Stephen Roth, Stefan Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2505.09246)  

**Abstract**: In many real-world settings, machine learning models and interactive systems have access to both structured knowledge, e.g., knowledge graphs or tables, and unstructured content, e.g., natural language documents. However, most rely on either. Semi-Structured Knowledge Bases (SKBs) bridge this gap by linking unstructured content to nodes within structured data, thereby enabling new strategies for knowledge access and use. In this work, we present FocusedRetriever, a modular SKB-based framework for multi-hop question answering. It integrates components (VSS-based entity search, LLM-based generation of Cypher queries and pairwise re-ranking) in a way that enables it to outperform state-of-the-art methods across all three STaRK benchmark test sets, covering diverse domains and multiple performance metrics. The average first-hit rate exceeds that of the second-best method by 25.7%. FocusedRetriever leverages (1) the capacity of Large Language Models (LLMs) to extract relational facts and entity attributes from unstructured text, (2) node set joins to filter answer candidates based on these extracted triplets and constraints, (3) vector similarity search to retrieve and rank relevant unstructured content, and (4) the contextual capabilities of LLMs to finally rank the top-k answers. For generality, we only incorporate base LLMs in FocusedRetriever in our evaluation. However, our analysis of intermediate results highlights several opportunities for further upgrades including finetuning. The source code is publicly available at this https URL . 

---
# Ornithologist: Towards Trustworthy "Reasoning" about Central Bank Communications 

**Authors**: Dominic Zaun Eu Jones  

**Link**: [PDF](https://arxiv.org/pdf/2505.09083)  

**Abstract**: I develop Ornithologist, a weakly-supervised textual classification system and measure the hawkishness and dovishness of central bank text. Ornithologist uses ``taxonomy-guided reasoning'', guiding a large language model with human-authored decision trees. This increases the transparency and explainability of the system and makes it accessible to non-experts. It also reduces hallucination risk. Since it requires less supervision than traditional classification systems, it can more easily be applied to other problems or sources of text (e.g. news) without much modification. Ornithologist measurements of hawkishness and dovishness of RBA communication carry information about the future of the cash rate path and of market expectations. 

---
# Improving the Reliability of LLMs: Combining CoT, RAG, Self-Consistency, and Self-Verification 

**Authors**: Adarsh Kumar, Hwiyoon Kim, Jawahar Sai Nathani, Neil Roy  

**Link**: [PDF](https://arxiv.org/pdf/2505.09031)  

**Abstract**: Hallucination, where large language models (LLMs) generate confident but incorrect or irrelevant information, remains a key limitation in their application to complex, open-ended tasks. Chain-of-thought (CoT) prompting has emerged as a promising method for improving multistep reasoning by guiding models through intermediate steps. However, CoT alone does not fully address the hallucination problem. In this work, we investigate how combining CoT with retrieval-augmented generation (RAG), as well as applying self-consistency and self-verification strategies, can reduce hallucinations and improve factual accuracy. By incorporating external knowledge sources during reasoning and enabling models to verify or revise their own outputs, we aim to generate more accurate and coherent responses. We present a comparative evaluation of baseline LLMs against CoT, CoT+RAG, self-consistency, and self-verification techniques. Our results highlight the effectiveness of each method and identify the most robust approach for minimizing hallucinations while preserving fluency and reasoning depth. 

---
# Automated Meta Prompt Engineering for Alignment with the Theory of Mind 

**Authors**: Aaron Baughman, Rahul Agarwal, Eduardo Morales, Gozde Akay  

**Link**: [PDF](https://arxiv.org/pdf/2505.09024)  

**Abstract**: We introduce a method of meta-prompting that jointly produces fluent text for complex tasks while optimizing the similarity of neural states between a human's mental expectation and a Large Language Model's (LLM) neural processing. A technique of agentic reinforcement learning is applied, in which an LLM as a Judge (LLMaaJ) teaches another LLM, through in-context learning, how to produce content by interpreting the intended and unintended generated text traits. To measure human mental beliefs around content production, users modify long form AI-generated text articles before publication at the US Open 2024 tennis Grand Slam. Now, an LLMaaJ can solve the Theory of Mind (ToM) alignment problem by anticipating and including human edits within the creation of text from an LLM. Throughout experimentation and by interpreting the results of a live production system, the expectations of human content reviewers had 100% of alignment with AI 53.8% of the time with an average iteration count of 4.38. The geometric interpretation of content traits such as factualness, novelty, repetitiveness, and relevancy over a Hilbert vector space combines spatial volume (all trait importance) with vertices alignment (individual trait relevance) enabled the LLMaaJ to optimize on Human ToM. This resulted in an increase in content quality by extending the coverage of tennis action. Our work that was deployed at the US Open 2024 has been used across other live events within sports and entertainment. 

---
# Prioritizing Image-Related Tokens Enhances Vision-Language Pre-Training 

**Authors**: Yangyi Chen, Hao Peng, Tong Zhang, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.08971)  

**Abstract**: In standard large vision-language models (LVLMs) pre-training, the model typically maximizes the joint probability of the caption conditioned on the image via next-token prediction (NTP); however, since only a small subset of caption tokens directly relates to the visual content, this naive NTP unintentionally fits the model to noise and increases the risk of hallucination. We present PRIOR, a simple vision-language pre-training approach that addresses this issue by prioritizing image-related tokens through differential weighting in the NTP loss, drawing from the importance sampling framework. PRIOR introduces a reference model-a text-only large language model (LLM) trained on the captions without image inputs, to weight each token based on its probability for LVLMs training. Intuitively, tokens that are directly related to the visual inputs are harder to predict without the image and thus receive lower probabilities from the text-only reference LLM. During training, we implement a token-specific re-weighting term based on the importance scores to adjust each token's loss. We implement PRIOR in two distinct settings: LVLMs with visual encoders and LVLMs without visual encoders. We observe 19% and 8% average relative improvement, respectively, on several vision-language benchmarks compared to NTP. In addition, PRIOR exhibits superior scaling properties, as demonstrated by significantly higher scaling coefficients, indicating greater potential for performance gains compared to NTP given increasing compute and data. 

---
# ForeCite: Adapting Pre-Trained Language Models to Predict Future Citation Rates of Academic Papers 

**Authors**: Gavin Hull, Alex Bihlo  

**Link**: [PDF](https://arxiv.org/pdf/2505.08941)  

**Abstract**: Predicting the future citation rates of academic papers is an important step toward the automation of research evaluation and the acceleration of scientific progress. We present $\textbf{ForeCite}$, a simple but powerful framework to append pre-trained causal language models with a linear head for average monthly citation rate prediction. Adapting transformers for regression tasks, ForeCite achieves a test correlation of $\rho = 0.826$ on a curated dataset of 900K+ biomedical papers published between 2000 and 2024, a 27-point improvement over the previous state-of-the-art. Comprehensive scaling-law analysis reveals consistent gains across model sizes and data volumes, while temporal holdout experiments confirm practical robustness. Gradient-based saliency heatmaps suggest a potentially undue reliance on titles and abstract texts. These results establish a new state-of-the-art in forecasting the long-term influence of academic research and lay the groundwork for the automated, high-fidelity evaluation of scientific contributions. 

---
# Behind Maya: Building a Multilingual Vision Language Model 

**Authors**: Nahid Alam, Karthik Reddy Kanjula, Surya Guthikonda, Timothy Chung, Bala Krishna S Vegesna, Abhipsha Das, Anthony Susevski, Ryan Sze-Yin Chan, S M Iftekhar Uddin, Shayekh Bin Islam, Roshan Santhosh, Snegha A, Drishti Sharma, Chen Liu, Isha Chaturvedi, Genta Indra Winata, Ashvanth.S, Snehanshu Mukherjee, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2505.08910)  

**Abstract**: In recent times, we have seen a rapid development of large Vision-Language Models (VLMs). They have shown impressive results on academic benchmarks, primarily in widely spoken languages but lack performance on low-resource languages and varied cultural contexts. To address these limitations, we introduce Maya, an open-source Multilingual VLM. Our contributions are: 1) a multilingual image-text pretraining dataset in eight languages, based on the LLaVA pretraining dataset; and 2) a multilingual image-text model supporting these languages, enhancing cultural and linguistic comprehension in vision-language tasks. Code available at this https URL. 

---
# Grounding Synthetic Data Evaluations of Language Models in Unsupervised Document Corpora 

**Authors**: Michael Majurski, Cynthia Matuszek  

**Link**: [PDF](https://arxiv.org/pdf/2505.08905)  

**Abstract**: Language Models (LMs) continue to advance, improving response quality and coherence. Given Internet-scale training datasets, LMs have likely encountered much of what users might ask them to generate in some form during their training. A plethora of evaluation benchmarks have been constructed to assess model quality, response appropriateness, and reasoning capabilities. However, the human effort required for benchmark construction is limited and being rapidly outpaced by the size and scope of the models under evaluation. Additionally, having humans build a benchmark for every possible domain of interest is impractical. Therefore, we propose a methodology for automating the construction of fact-based synthetic data model evaluations grounded in document populations. This work leverages those very same LMs to evaluate domain-specific knowledge automatically, using only grounding documents (e.g., a textbook) as input. This synthetic data benchmarking approach corresponds well with human curated questions with a Spearman ranking correlation of 0.96 and a benchmark evaluation Pearson accuracy correlation of 0.79. This novel tool supports generating both multiple choice and open-ended synthetic data questions to gain diagnostic insight of LM capability. We apply this methodology to evaluate model performance on a recent relevant arXiv preprint, discovering a surprisingly strong performance from Gemma3 models. 

---
# Performance Gains of LLMs With Humans in a World of LLMs Versus Humans 

**Authors**: Lucas McCullum, Pelagie Ami Agassi, Leo Anthony Celi, Daniel K. Ebner, Chrystinne Oliveira Fernandes, Rachel S. Hicklen, Mkliwa Koumbia, Lisa Soleymani Lehmann, David Restrepo  

**Link**: [PDF](https://arxiv.org/pdf/2505.08902)  

**Abstract**: Currently, a considerable research effort is devoted to comparing LLMs to a group of human experts, where the term "expert" is often ill-defined or variable, at best, in a state of constantly updating LLM releases. Without proper safeguards in place, LLMs will threaten to cause harm to the established structure of safe delivery of patient care which has been carefully developed throughout history to keep the safety of the patient at the forefront. A key driver of LLM innovation is founded on community research efforts which, if continuing to operate under "humans versus LLMs" principles, will expedite this trend. Therefore, research efforts moving forward must focus on effectively characterizing the safe use of LLMs in clinical settings that persist across the rapid development of novel LLM models. In this communication, we demonstrate that rather than comparing LLMs to humans, there is a need to develop strategies enabling efficient work of humans with LLMs in an almost symbiotic manner. 

---
# LibVulnWatch: A Deep Assessment Agent System and Leaderboard for Uncovering Hidden Vulnerabilities in Open-Source AI Libraries 

**Authors**: Zekun Wu, Seonglae Cho, Umar Mohammed, Cristian Munoz, Kleyton Costa, Xin Guan, Theo King, Ze Wang, Emre Kazim, Adriano Koshiyama  

**Link**: [PDF](https://arxiv.org/pdf/2505.08842)  

**Abstract**: Open-source AI libraries are foundational to modern AI systems but pose significant, underexamined risks across security, licensing, maintenance, supply chain integrity, and regulatory compliance. We present LibVulnWatch, a graph-based agentic assessment framework that performs deep, source-grounded evaluations of these libraries. Built on LangGraph, the system coordinates a directed acyclic graph of specialized agents to extract, verify, and quantify risk using evidence from trusted sources such as repositories, documentation, and vulnerability databases. LibVulnWatch generates reproducible, governance-aligned scores across five critical domains, publishing them to a public leaderboard for longitudinal ecosystem monitoring. Applied to 20 widely used libraries, including ML frameworks, LLM inference engines, and agent orchestration tools, our system covers up to 88% of OpenSSF Scorecard checks while uncovering up to 19 additional risks per library. These include critical Remote Code Execution (RCE) vulnerabilities, absent Software Bills of Materials (SBOMs), licensing constraints, undocumented telemetry, and widespread gaps in regulatory documentation and auditability. By translating high-level governance principles into practical, verifiable metrics, LibVulnWatch advances technical AI governance with a scalable, transparent mechanism for continuous supply chain risk assessment and informed library selection. 

---
# An Extra RMSNorm is All You Need for Fine Tuning to 1.58 Bits 

**Authors**: Cody Steinmetz, Gavin Childress, Aaron Herbst, Gavin Jones, Jasdeep Singh, Eli Vang, Keagan Weinstock  

**Link**: [PDF](https://arxiv.org/pdf/2505.08823)  

**Abstract**: Large language models (LLMs) have transformed natural-language processing, yet their scale makes real-world deployment costly. Post-training quantization reduces memory and computation but often degrades accuracy, while quantization-aware training can recover performance at the cost of extra training. Pushing quantization to the ternary (2-bit) regime yields even larger savings but is notoriously unstable. Building on recent work showing that a bias-free, RMS-normalized Transformer with straight-through estimation can reach 1.58-bit precision, we demonstrate that simply inserting RMS normalization before every linear projection and applying a gradual, layer-wise quantization schedule stably fine-tunes full-precision checkpoints into ternary LLMs. Our approach matches or surpasses more elaborate knowledge-distillation pipelines on standard language-modeling benchmarks without adding model complexity. These results indicate that careful normalization alone can close much of the accuracy gap between ternary and full-precision LLMs, making ultra-low-bit inference practical. 

---
# The Geometry of Meaning: Perfect Spacetime Representations of Hierarchical Structures 

**Authors**: Andres Anabalon, Hugo Garces, Julio Oliva, Jose Cifuentes  

**Link**: [PDF](https://arxiv.org/pdf/2505.08795)  

**Abstract**: We show that there is a fast algorithm that embeds hierarchical structures in three-dimensional Minkowski spacetime. The correlation of data ends up purely encoded in the causal structure. Our model relies solely on oriented token pairs -- local hierarchical signals -- with no access to global symbolic structure. We apply our method to the corpus of \textit{WordNet}. We provide a perfect embedding of the mammal sub-tree including ambiguities (more than one hierarchy per node) in such a way that the hierarchical structures get completely codified in the geometry and exactly reproduce the ground-truth. We extend this to a perfect embedding of the maximal unambiguous subset of the \textit{WordNet} with 82{,}115 noun tokens and a single hierarchy per token. We introduce a novel retrieval mechanism in which causality, not distance, governs hierarchical access. Our results seem to indicate that all discrete data has a perfect geometrical representation that is three-dimensional. The resulting embeddings are nearly conformally invariant, indicating deep connections with general relativity and field theory. These results suggest that concepts, categories, and their interrelations, namely hierarchical meaning itself, is geometric. 

---
