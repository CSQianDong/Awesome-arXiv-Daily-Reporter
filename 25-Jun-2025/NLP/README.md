# MAM: Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis via Role-Specialized Collaboration 

**Authors**: Yucheng Zhou, Lingran Song, Jianbing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19835)  

**Abstract**: Recent advancements in medical Large Language Models (LLMs) have showcased their powerful reasoning and diagnostic capabilities. Despite their success, current unified multimodal medical LLMs face limitations in knowledge update costs, comprehensiveness, and flexibility. To address these challenges, we introduce the Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis (MAM). Inspired by our empirical findings highlighting the benefits of role assignment and diagnostic discernment in LLMs, MAM decomposes the medical diagnostic process into specialized roles: a General Practitioner, Specialist Team, Radiologist, Medical Assistant, and Director, each embodied by an LLM-based agent. This modular and collaborative framework enables efficient knowledge updates and leverages existing medical LLMs and knowledge bases. Extensive experimental evaluations conducted on a wide range of publicly accessible multimodal medical datasets, incorporating text, image, audio, and video modalities, demonstrate that MAM consistently surpasses the performance of modality-specific LLMs. Notably, MAM achieves significant performance improvements ranging from 18% to 365% compared to baseline models. Our code is released at this https URL. 

---
# How Effectively Can BERT Models Interpret Context and Detect Bengali Communal Violent Text? 

**Authors**: Abdullah Khondoker, Enam Ahmed Taufik, Md. Iftekhar Islam Tashik, S M Ishtiak Mahmud, Farig Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2506.19831)  

**Abstract**: The spread of cyber hatred has led to communal violence, fueling aggression and conflicts between various religious, ethnic, and social groups, posing a significant threat to social harmony. Despite its critical importance, the classification of communal violent text remains an underexplored area in existing research. This study aims to enhance the accuracy of detecting text that incites communal violence, focusing specifically on Bengali textual data sourced from social media platforms. We introduce a fine-tuned BanglaBERT model tailored for this task, achieving a macro F1 score of 0.60. To address the issue of data imbalance, our dataset was expanded by adding 1,794 instances, which facilitated the development and evaluation of a fine-tuned ensemble model. This ensemble model demonstrated an improved performance, achieving a macro F1 score of 0.63, thus highlighting its effectiveness in this domain. In addition to quantitative performance metrics, qualitative analysis revealed instances where the models struggled with context understanding, leading to occasional misclassifications, even when predictions were made with high confidence. Through analyzing the cosine similarity between words, we identified certain limitations in the pre-trained BanglaBERT models, particularly in their ability to distinguish between closely related communal and non-communal terms. To further interpret the model's decisions, we applied LIME, which helped to uncover specific areas where the model struggled in understanding context, contributing to errors in classification. These findings highlight the promise of NLP and interpretability tools in reducing online communal violence. Our work contributes to the growing body of research in communal violence detection and offers a foundation for future studies aiming to refine these techniques for better accuracy and societal impact. 

---
# Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study 

**Authors**: Yuqi Zhu, Yi Zhong, Jintian Zhang, Ziheng Zhang, Shuofei Qiao, Yujie Luo, Lun Du, Da Zheng, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19794)  

**Abstract**: Large Language Models (LLMs) hold promise in automating data analysis tasks, yet open-source models face significant limitations in these kinds of reasoning-intensive scenarios. In this work, we investigate strategies to enhance the data analysis capabilities of open-source LLMs. By curating a seed dataset of diverse, realistic scenarios, we evaluate models across three dimensions: data understanding, code generation, and strategic planning. Our analysis reveals three key findings: (1) Strategic planning quality serves as the primary determinant of model performance; (2) Interaction design and task complexity significantly influence reasoning capabilities; (3) Data quality demonstrates a greater impact than diversity in achieving optimal performance. We leverage these insights to develop a data synthesis methodology, demonstrating significant improvements in open-source LLMs' analytical reasoning capabilities. 

---
# SRFT: A Single-Stage Method with Supervised and Reinforcement Fine-Tuning for Reasoning 

**Authors**: Yuqian Fu, Tinghong Chen, Jiajun Chai, Xihuai Wang, Songjun Tu, Guojun Yin, Wei Lin, Qichao Zhang, Yuanheng Zhu, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.19767)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress in reasoning tasks, yet the optimal integration of Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) remains a fundamental challenge. Through comprehensive analysis of token distributions, learning dynamics, and integration mechanisms from entropy-based perspectives, we reveal key differences between these paradigms: SFT induces coarse-grained global changes to LLM policy distributions, while RL performs fine-grained selective optimizations, with entropy serving as a critical indicator of training effectiveness. Building on these observations, we propose Supervised Reinforcement Fine-Tuning (SRFT), a single-stage method that unifies both fine-tuning paradigms through entropy-aware weighting mechanisms. Our approach simultaneously applies SFT and RL to directly optimize the LLM using demonstrations and self-exploration rollouts rather than through two-stage sequential methods. Extensive experiments show that SRFT achieves 59.1% average accuracy, outperforming zero-RL methods by 9.0% on five mathematical reasoning benchmarks and 10.9% on three out-of-distribution benchmarks. 

---
# Accurate, fast, cheap: Choose three. Replacing Multi-Head-Attention with Bidirectional Recurrent Attention for Long-Form ASR 

**Authors**: Martin Ratajczak, Jean-Philippe Robichaud, Jennifer Drexler Fox  

**Link**: [PDF](https://arxiv.org/pdf/2506.19761)  

**Abstract**: Long-form speech recognition is an application area of increasing research focus. ASR models based on multi-head attention (MHA) are ill-suited to long-form ASR because of their quadratic complexity in sequence length. We build on recent work that has investigated linear complexity recurrent attention (RA) layers for ASR. We find that bidirectional RA layers can match the accuracy of MHA for both short- and long-form applications. We present a strong limited-context attention (LCA) baseline, and show that RA layers are just as accurate while being more efficient. We develop a long-form training paradigm which further improves RA performance, leading to better accuracy than LCA with 44% higher throughput. We also present Direction Dropout, a novel regularization method that improves accuracy, provides fine-grained control of the accuracy/throughput trade-off of bidirectional RA, and enables a new alternating directions decoding mode with even higher throughput. 

---
# Arabic Dialect Classification using RNNs, Transformers, and Large Language Models: A Comparative Analysis 

**Authors**: Omar A.Essameldin, Ali O.Elbeih, Wael H.Gomaa, Wael F.Elsersy  

**Link**: [PDF](https://arxiv.org/pdf/2506.19753)  

**Abstract**: The Arabic language is among the most popular languages in the world with a huge variety of dialects spoken in 22 countries. In this study, we address the problem of classifying 18 Arabic dialects of the QADI dataset of Arabic tweets. RNN models, Transformer models, and large language models (LLMs) via prompt engineering are created and tested. Among these, MARBERTv2 performed best with 65% accuracy and 64% F1-score. Through the use of state-of-the-art preprocessing techniques and the latest NLP models, this paper identifies the most significant linguistic issues in Arabic dialect identification. The results corroborate applications like personalized chatbots that respond in users' dialects, social media monitoring, and greater accessibility for Arabic communities. 

---
# Evaluating Rare Disease Diagnostic Performance in Symptom Checkers: A Synthetic Vignette Simulation Approach 

**Authors**: Takashi Nishibayashi, Seiji Kanazawa, Kumpei Yamada  

**Link**: [PDF](https://arxiv.org/pdf/2506.19750)  

**Abstract**: Background: Symptom Checkers (SCs) provide users with personalized medical information. To prevent performance degradation from algorithm updates, SC developers must evaluate diagnostic performance changes for individual diseases before deployment. However, acquiring sufficient evaluation data for rare diseases is difficult, and manually creating numerous clinical vignettes is costly and impractical. Objective: This study proposes and validates a novel Synthetic Vignette Simulation Approach to evaluate diagnostic performance changes for individual rare diseases following SC algorithm updates. Methods: We used disease-phenotype annotations from the Human Phenotype Ontology (HPO), a knowledge database for rare diseases, to generate synthetic vignettes. With these, we simulated SC interviews to estimate the impact of algorithm updates on real-world diagnostic performance. The method's effectiveness was evaluated retrospectively by comparing estimated values with actual metric changes using the R 2(R-squared) coefficient. Results: The experiment included eight past SC algorithm updates. For updates on diseases with frequency information in HPO (n=5), the R^2 for recall@8 change was 0.831 (p=0.031), and for precision@8 change, it was 0.78 (p=0.047), indicating the method can predict post-deployment performance. In contrast, large prediction errors occurred for diseases without frequency information (n=3), highlighting its importance. The manual effort to map HPO phenotypes to SC symptoms was approximately 2 hours per disease. Conclusions: Our method enables pre-deployment evaluation of SC algorithm changes for individual rare diseases using a publicly available, expert-created knowledge base. This transparent and low-cost approach allows developers to efficiently improve diagnostic performance for rare diseases, potentially enhancing support for early diagnosis. 

---
# Breaking Barriers: Do Reinforcement Post Training Gains Transfer To Unseen Domains? 

**Authors**: Chuxuan Hu, Yuxuan Zhu, Antony Kellermann, Caleb Biddulph, Suppakit Waiwitlikhit, Jason Benn, Daniel Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19733)  

**Abstract**: Reinforcement post training (RPT) has recently shown promise in improving the reasoning abilities of large language models (LLMs). However, it remains unclear how well these improvements generalize to new domains, as prior work evaluates RPT models on data from the same domains used for fine-tuning. To understand the generalizability of RPT, we conduct two studies. (1) Observational: We compare a wide range of open-weight RPT models against their corresponding base models across multiple domains, including both seen and unseen domains in their fine-tuning data. (2) Interventional: we fine-tune LLMs with RPT on single domains and evaluate their performance across multiple domains. Both studies converge on the same conclusion that, although RPT brings substantial gains on tasks similar to the fine-tuning data, the gains generalize inconsistently and can vanish on domains with different reasoning patterns. 

---
# Tailored Conversations beyond LLMs: A RL-Based Dialogue Manager 

**Authors**: Lucie Galland, Catherine Pelachaud, Florian Pecune  

**Link**: [PDF](https://arxiv.org/pdf/2506.19652)  

**Abstract**: In this work, we propose a novel framework that integrates large language models (LLMs) with an RL-based dialogue manager for open-ended dialogue with a specific goal. By leveraging hierarchical reinforcement learning to model the structured phases of dialogue and employ meta-learning to enhance adaptability across diverse user profiles, our approach enhances adaptability and efficiency, enabling the system to learn from limited data, transition fluidly between dialogue phases, and personalize responses to heterogeneous patient needs. We apply our framework to Motivational Interviews, aiming to foster behavior change, and demonstrate that the proposed dialogue manager outperforms a state-of-the-art LLM baseline in terms of reward, showing a potential benefit of conditioning LLMs to create open-ended dialogue systems with specific goals. 

---
# Correcting Hallucinations in News Summaries: Exploration of Self-Correcting LLM Methods with External Knowledge 

**Authors**: Juraj Vladika, Ihsan Soydemir, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2506.19607)  

**Abstract**: While large language models (LLMs) have shown remarkable capabilities to generate coherent text, they suffer from the issue of hallucinations -- factually inaccurate statements. Among numerous approaches to tackle hallucinations, especially promising are the self-correcting methods. They leverage the multi-turn nature of LLMs to iteratively generate verification questions inquiring additional evidence, answer them with internal or external knowledge, and use that to refine the original response with the new corrections. These methods have been explored for encyclopedic generation, but less so for domains like news summarization. In this work, we investigate two state-of-the-art self-correcting systems by applying them to correct hallucinated summaries using evidence from three search engines. We analyze the results and provide insights into systems' performance, revealing interesting practical findings on the benefits of search engine snippets and few-shot prompts, as well as high alignment of G-Eval and human evaluation. 

---
# Social Hatred: Efficient Multimodal Detection of Hatemongers 

**Authors**: Tom Marzea, Abraham Israeli, Oren Tsur  

**Link**: [PDF](https://arxiv.org/pdf/2506.19603)  

**Abstract**: Automatic detection of online hate speech serves as a crucial step in the detoxification of the online discourse. Moreover, accurate classification can promote a better understanding of the proliferation of hate as a social phenomenon. While most prior work focus on the detection of hateful utterances, we argue that focusing on the user level is as important, albeit challenging. In this paper we consider a multimodal aggregative approach for the detection of hate-mongers, taking into account the potentially hateful texts, user activity, and the user network. Evaluating our method on three unique datasets X (Twitter), Gab, and Parler we show that processing a user's texts in her social context significantly improves the detection of hate mongers, compared to previously used text and graph-based methods. We offer comprehensive set of results obtained in different experimental settings as well as qualitative analysis of illustrative cases. Our method can be used to improve the classification of coded messages, dog-whistling, and racial gas-lighting, as well as to inform intervention measures. Moreover, we demonstrate that our multimodal approach performs well across very different content platforms and over large datasets and networks. 

---
# ECCoT: A Framework for Enhancing Effective Cognition via Chain of Thought in Large Language Model 

**Authors**: Zhenke Duan, Jiqun Pan, Jiani Tu, Xiaoyi Wang, Yanqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19599)  

**Abstract**: In the era of large-scale artificial intelligence, Large Language Models (LLMs) have made significant strides in natural language processing. However, they often lack transparency and generate unreliable outputs, raising concerns about their interpretability. To address this, the Chain of Thought (CoT) prompting method structures reasoning into step-by-step deductions. Yet, not all reasoning chains are valid, and errors can lead to unreliable conclusions. We propose ECCoT, an End-to-End Cognitive Chain of Thought Validation Framework, to evaluate and refine reasoning chains in LLMs. ECCoT integrates the Markov Random Field-Embedded Topic Model (MRF-ETM) for topic-aware CoT generation and Causal Sentence-BERT (CSBert) for causal reasoning alignment. By filtering ineffective chains using structured ordering statistics, ECCoT improves interpretability, reduces biases, and enhances the trustworthiness of LLM-based decision-making. Key contributions include the introduction of ECCoT, MRF-ETM for topic-driven CoT generation, and CSBert for causal reasoning enhancement. Code is released at: this https URL. 

---
# Has Machine Translation Evaluation Achieved Human Parity? The Human Reference and the Limits of Progress 

**Authors**: Lorenzo Proietti, Stefano Perrella, Roberto Navigli  

**Link**: [PDF](https://arxiv.org/pdf/2506.19571)  

**Abstract**: In Machine Translation (MT) evaluation, metric performance is assessed based on agreement with human judgments. In recent years, automatic metrics have demonstrated increasingly high levels of agreement with humans. To gain a clearer understanding of metric performance and establish an upper bound, we incorporate human baselines in the MT meta-evaluation, that is, the assessment of MT metrics' capabilities. Our results show that human annotators are not consistently superior to automatic metrics, with state-of-the-art metrics often ranking on par with or higher than human baselines. Despite these findings suggesting human parity, we discuss several reasons for caution. Finally, we explore the broader implications of our results for the research field, asking: Can we still reliably measure improvements in MT evaluation? With this work, we aim to shed light on the limits of our ability to measure progress in the field, fostering discussion on an issue that we believe is crucial to the entire MT evaluation community. 

---
# RCStat: A Statistical Framework for using Relative Contextualization in Transformers 

**Authors**: Debabrata Mahapatra, Shubham Agarwal, Apoorv Saxena, Subrata Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2506.19549)  

**Abstract**: Prior work on input-token importance in auto-regressive transformers has relied on Softmax-normalized attention weights, which obscure the richer structure of pre-Softmax query-key logits. We introduce RCStat, a statistical framework that harnesses raw attention logits via Relative Contextualization (RC), a random variable measuring contextual alignment between token segments, and derive an efficient upper bound for RC. We demonstrate two applications: (i) Key-Value compression, where RC-based thresholds drive adaptive key-value eviction for substantial cache reduction with minimal quality loss; and (ii) Attribution, where RC yields higher-fidelity token-, sentence-, and chunk-level explanations than post-Softmax methods. Across question answering, summarization, and attribution benchmarks, RCStat achieves significant empirical gains, delivering state-of-the-art compression and attribution performance without any model retraining. 

---
# Health Sentinel: An AI Pipeline For Real-time Disease Outbreak Detection 

**Authors**: Devesh Pant, Rishi Raj Grandhe, Vipin Samaria, Mukul Paul, Sudhir Kumar, Saransh Khanna, Jatin Agrawal, Jushaan Singh Kalra, Akhil VSSG, Satish V Khalikar, Vipin Garg, Himanshu Chauhan, Pranay Verma, Neha Khandelwal, Soma S Dhavala, Minesh Mathew  

**Link**: [PDF](https://arxiv.org/pdf/2506.19548)  

**Abstract**: Early detection of disease outbreaks is crucial to ensure timely intervention by the health authorities. Due to the challenges associated with traditional indicator-based surveillance, monitoring informal sources such as online media has become increasingly popular. However, owing to the number of online articles getting published everyday, manual screening of the articles is impractical. To address this, we propose Health Sentinel. It is a multi-stage information extraction pipeline that uses a combination of ML and non-ML methods to extract events-structured information concerning disease outbreaks or other unusual health events-from online articles. The extracted events are made available to the Media Scanning and Verification Cell (MSVC) at the National Centre for Disease Control (NCDC), Delhi for analysis, interpretation and further dissemination to local agencies for timely intervention. From April 2022 till date, Health Sentinel has processed over 300 million news articles and identified over 95,000 unique health events across India of which over 3,500 events were shortlisted by the public health experts at NCDC as potential outbreaks. 

---
# KnowMap: Efficient Knowledge-Driven Task Adaptation for LLMs 

**Authors**: Kelin Fu, Kaigui Bian  

**Link**: [PDF](https://arxiv.org/pdf/2506.19527)  

**Abstract**: While Large Language Models (LLMs) possess significant capabilities in open-world agent tasks, they also face challenges in rapidly adapting to new, specialized tasks due to their reliance on static pre-trained knowledge. Traditional methods such as fine-tuning are often costly, data-intensive, and may lead to "catastrophic forgetting." Therefore, we present KnowMap, a novel approach that dynamically constructs a knowledge base from environmental and experiential data. KnowMap fine-tunes a small knowledge-embedding model to equip a larger LLM with valuable task-specific knowledge. Our experiments on the ScienceWorld benchmark demonstrate 17.71% improvement for the performance of gpt-4-turbo model. KnowMap not only provides an efficient and effective means for LLM task-adapting, but also highlights how integrating environmental and experiential knowledge can enhance LLMs' reasoning capabilities. 

---
# Automatic Posology Structuration : What role for LLMs? 

**Authors**: Natalia Bobkova, Laura Zanella-Calzada, Anyes Tafoughalt, Raphaël Teboul, François Plesse, Félix Gaschi  

**Link**: [PDF](https://arxiv.org/pdf/2506.19525)  

**Abstract**: Automatically structuring posology instructions is essential for improving medication safety and enabling clinical decision support. In French prescriptions, these instructions are often ambiguous, irregular, or colloquial, limiting the effectiveness of classic ML pipelines. We explore the use of Large Language Models (LLMs) to convert free-text posologies into structured formats, comparing prompt-based methods and fine-tuning against a "pre-LLM" system based on Named Entity Recognition and Linking (NERL). Our results show that while prompting improves performance, only fine-tuned LLMs match the accuracy of the baseline. Through error analysis, we observe complementary strengths: NERL offers structural precision, while LLMs better handle semantic nuances. Based on this, we propose a hybrid pipeline that routes low-confidence cases from NERL (<0.8) to the LLM, selecting outputs based on confidence scores. This strategy achieves 91% structuration accuracy while minimizing latency and compute. Our results show that this hybrid approach improves structuration accuracy while limiting computational cost, offering a scalable solution for real-world clinical use. 

---
# heiDS at ArchEHR-QA 2025: From Fixed-k to Query-dependent-k for Retrieval Augmented Generation 

**Authors**: Ashish Chouhan, Michael Gertz  

**Link**: [PDF](https://arxiv.org/pdf/2506.19512)  

**Abstract**: This paper presents the approach of our team called heiDS for the ArchEHR-QA 2025 shared task. A pipeline using a retrieval augmented generation (RAG) framework is designed to generate answers that are attributed to clinical evidence from the electronic health records (EHRs) of patients in response to patient-specific questions. We explored various components of a RAG framework, focusing on ranked list truncation (RLT) retrieval strategies and attribution approaches. Instead of using a fixed top-k RLT retrieval strategy, we employ a query-dependent-k retrieval strategy, including the existing surprise and autocut methods and two new methods proposed in this work, autocut* and elbow. The experimental results show the benefits of our strategy in producing factual and relevant answers when compared to a fixed-$k$. 

---
# AnTKV: Anchor Token-Aware Sub-Bit Vector Quantization for KV Cache in Large Language Models 

**Authors**: Zeyu Li, Chuanfu Xiao, Yang Wang, Xiang Liu, Zhenheng Tang, Baotong Lu, Mao Yang, Xinyu Chen, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19505)  

**Abstract**: Quantization has emerged as an effective and lightweight solution to reduce the memory footprint of the KV cache in Large Language Models (LLMs). Nevertheless, minimizing the performance degradation caused by ultra-low-bit KV cache quantization remains a significant challenge. We observe that quantizing the KV cache of different tokens has varying impacts on the quality of attention outputs. To systematically investigate this phenomenon, we perform forward error propagation analysis on attention and propose the Anchor Score (AnS) that quantifies the sensitivity of each token's KV cache to quantization-induced error. Our analysis reveals significant disparities in AnS across tokens, suggesting that preserving a small subset with full precision (FP16) of high-AnS tokens can greatly mitigate accuracy loss in aggressive quantization scenarios. Based on this insight, we introduce AnTKV, a novel framework that leverages Anchor Token-aware Vector Quantization to compress the KV cache. Furthermore, to support efficient deployment, we design and develop a triton kernel that is fully compatible with FlashAttention, enabling fast online Anchor Token selection. AnTKV enables LLaMA-3-8B to handle context lengths up to 840K tokens on a single 80GB A100 GPU, while achieving up to 3.5x higher decoding throughput compared to the FP16 baseline. Our experiment results demonstrate that AnTKV matches or outperforms prior works such as KIVI, SKVQ, KVQuant, and CQ under 4-bit settings. More importantly, AnTKV achieves significantly lower perplexity under ultra-low-bit quantization on Mistral-7B, with only 6.32 at 1-bit and 8.87 at 0.375-bit, compared to the FP16 baseline of 4.73. 

---
# Is Long-to-Short a Free Lunch? Investigating Inconsistency and Reasoning Efficiency in LRMs 

**Authors**: Shu Yang, Junchao Wu, Xuansheng Wu, Derek Wong, Ninhao Liu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19492)  

**Abstract**: Large Reasoning Models (LRMs) have achieved remarkable performance on complex tasks by engaging in extended reasoning before producing final answers, yet this strength introduces the risk of overthinking, where excessive token generation occurs even for simple tasks. While recent work in efficient reasoning seeks to reduce reasoning length while preserving accuracy, it remains unclear whether such optimization is truly a free lunch. Drawing on the intuition that compressing reasoning may reduce the robustness of model responses and lead models to omit key reasoning steps, we investigate whether efficient reasoning strategies introduce behavioral inconsistencies. To systematically assess this, we introduce $ICBENCH$, a benchmark designed to measure inconsistency in LRMs across three dimensions: inconsistency across task settings (ITS), inconsistency between training objectives and learned behavior (TR-LB), and inconsistency between internal reasoning and self-explanations (IR-SE). Applying $ICBENCH$ to a range of open-source LRMs, we find that while larger models generally exhibit greater consistency than smaller ones, they all display widespread "scheming" behaviors, including self-disagreement, post-hoc rationalization, and the withholding of reasoning cues. Crucially, our results demonstrate that efficient reasoning strategies such as No-Thinking and Simple Token-Budget consistently increase all three defined types of inconsistency. These findings suggest that although efficient reasoning enhances token-level efficiency, further investigation is imperative to ascertain whether it concurrently introduces the risk of models evading effective supervision. 

---
# Dialogic Pedagogy for Large Language Models: Aligning Conversational AI with Proven Theories of Learning 

**Authors**: Russell Beale  

**Link**: [PDF](https://arxiv.org/pdf/2506.19484)  

**Abstract**: Large Language Models (LLMs) are rapidly transforming education by enabling rich conversational learning experiences. This article provides a comprehensive review of how LLM-based conversational agents are being used in higher education, with extensions to secondary and lifelong learning contexts. We synthesize existing literature on LLMs in education and theories of conversational and dialogic pedagogy - including Vygotsky's sociocultural learning (scaffolding and the Zone of Proximal Development), the Socratic method, and Laurillard's conversational framework - and examine how prompting strategies and retrieval-augmented generation (RAG) can align LLM behaviors with these pedagogical theories, and how it can support personalized, adaptive learning. We map educational theories to LLM capabilities, highlighting where LLM-driven dialogue supports established learning principles and where it challenges or falls short of traditional pedagogical assumptions. Notable gaps in applying prior theories to LLMs are identified, such as the models tendency to provide direct answers instead of fostering co-construction of knowledge, and the need to account for the constant availability and broad but non-human expertise of LLM tutors. In response, we propose practical strategies to better align LLM interactions with sound pedagogy - for example, designing prompts that encourage Socratic questioning, scaffolded guidance, and student reflection, as well as integrating retrieval mechanisms to ensure accuracy and contextual relevance. Our aim is to bridge the gap between educational theory and the emerging practice of AI-driven conversational learning, offering insights and tools for making LLM-based dialogues more educationally productive and theory-aligned. 

---
# Commonsense Generation and Evaluation for Dialogue Systems using Large Language Models 

**Authors**: Marcos Estecha-Garitagoitia, Chen Zhang, Mario Rodríguez-Cantelar, Luis Fernando D'Haro  

**Link**: [PDF](https://arxiv.org/pdf/2506.19483)  

**Abstract**: This paper provides preliminary results on exploring the task of performing turn-level data augmentation for dialogue system based on different types of commonsense relationships, and the automatic evaluation of the generated synthetic turns. The proposed methodology takes advantage of the extended knowledge and zero-shot capabilities of pretrained Large Language Models (LLMs) to follow instructions, understand contextual information, and their commonsense reasoning capabilities. The approach draws inspiration from methodologies like Chain-of-Thought (CoT), applied more explicitly to the task of prompt-based generation for dialogue-based data augmentation conditioned on commonsense attributes, and the automatic evaluation of the generated dialogues.
To assess the effectiveness of the proposed approach, first we extracted 200 randomly selected partial dialogues, from 5 different well-known dialogue datasets, and generate alternative responses conditioned on different event commonsense attributes. This novel dataset allows us to measure the proficiency of LLMs in generating contextually relevant commonsense knowledge, particularly up to 12 different specific ATOMIC [10] database relations. Secondly, we propose an evaluation framework to automatically detect the quality of the generated dataset inspired by the ACCENT [26] metric, which offers a nuanced approach to assess event commonsense. However, our method does not follow ACCENT's complex eventrelation tuple extraction process. Instead, we propose an instruction-based prompt for each commonsense attribute and use state-of-the-art LLMs to automatically detect the original attributes used when creating each augmented turn in the previous step.
Preliminary results suggest that our approach effectively harnesses LLMs capabilities for commonsense reasoning and evaluation in dialogue systems. 

---
# MuBench: Assessment of Multilingual Capabilities of Large Language Models Across 61 Languages 

**Authors**: Wenhan Han, Yifan Zhang, Zhixun Chen, Binbin Liu, Haobin Lin, Bingni Zhang, Taifeng Wang, Mykola Pechenizkiy, Meng Fang, Yin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.19468)  

**Abstract**: Multilingual large language models (LLMs) are advancing rapidly, with new models frequently claiming support for an increasing number of languages. However, existing evaluation datasets are limited and lack cross-lingual alignment, leaving assessments of multilingual capabilities fragmented in both language and skill coverage. To address this, we introduce MuBench, a benchmark covering 61 languages and evaluating a broad range of capabilities. We evaluate several state-of-the-art multilingual LLMs and find notable gaps between claimed and actual language coverage, particularly a persistent performance disparity between English and low-resource languages. Leveraging MuBench's alignment, we propose Multilingual Consistency (MLC) as a complementary metric to accuracy for analyzing performance bottlenecks and guiding model improvement. Finally, we pretrain a suite of 1.2B-parameter models on English and Chinese with 500B tokens, varying language ratios and parallel data proportions to investigate cross-lingual transfer dynamics. 

---
# Can Large Language Models Capture Human Annotator Disagreements? 

**Authors**: Jingwei Ni, Yu Fan, Vilém Zouhar, Donya Rooein, Alexander Hoyle, Mrinmaya Sachan, Markus Leippold, Dirk Hovy, Elliott Ash  

**Link**: [PDF](https://arxiv.org/pdf/2506.19467)  

**Abstract**: Human annotation variation (i.e., annotation disagreements) is common in NLP and often reflects important information such as task subjectivity and sample ambiguity. While Large Language Models (LLMs) are increasingly used for automatic annotation to reduce human effort, their evaluation often focuses on predicting the majority-voted "ground truth" labels. It is still unclear, however, whether these models also capture informative human annotation variation. Our work addresses this gap by extensively evaluating LLMs' ability to predict annotation disagreements without access to repeated human labels. Our results show that LLMs struggle with modeling disagreements, which can be overlooked by majority label-based evaluations. Notably, while RLVR-style (Reinforcement learning with verifiable rewards) reasoning generally boosts LLM performance, it degrades performance in disagreement prediction. Our findings highlight the critical need for evaluating and improving LLM annotators in disagreement modeling. Code and data at this https URL. 

---
# Learning to Disentangle Latent Reasoning Rules with Language VAEs: A Systematic Study 

**Authors**: Yingji Zhang, Marco Valentino, Danilo S. Carvalho, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2506.19418)  

**Abstract**: Incorporating explicit reasoning rules within the latent space of language models (LMs) offers a promising pathway to enhance generalisation, interpretability, and controllability. While current Transformer-based language models have shown strong performance on Natural Language Inference (NLI) tasks, they often rely on memorisation rather than rule-based inference. This work investigates how reasoning rules can be explicitly embedded and memorised within the LMs through Language Variational Autoencoders (VAEs). We propose a complete pipeline for learning reasoning rules within Transformer-based language VAEs. This pipeline encompasses three rule-based reasoning tasks, a supporting theoretical framework, and a practical end-to-end architecture. The experiment illustrates the following findings: Disentangled reasoning: Under explicit signal supervision, reasoning rules - viewed as functional mappings - can be disentangled within the encoder's parametric space. This separation results in distinct clustering of rules in the output feature space. Prior knowledge injection: injecting reasoning information into the Query enables the model to more effectively retrieve the stored value Value from memory based on Key. This approach offers a simple method for integrating prior knowledge into decoder-only language models. Performance bottleneck: In mathematical reasoning tasks using Qwen2.5(0.5B), increasing sample count doesn't improve performance beyond a point. Moreover, ffn layers are better than attention layers at preserving the separation of reasoning rules in the model's parameters. 

---
# Automated Detection of Pre-training Text in Black-box LLMs 

**Authors**: Ruihan Hu, Yu-Ming Shang, Jiankun Peng, Wei Luo, Yazhe Wang, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19399)  

**Abstract**: Detecting whether a given text is a member of the pre-training data of Large Language Models (LLMs) is crucial for ensuring data privacy and copyright protection. Most existing methods rely on the LLM's hidden information (e.g., model parameters or token probabilities), making them ineffective in the black-box setting, where only input and output texts are accessible. Although some methods have been proposed for the black-box setting, they rely on massive manual efforts such as designing complicated questions or instructions. To address these issues, we propose VeilProbe, the first framework for automatically detecting LLMs' pre-training texts in a black-box setting without human intervention. VeilProbe utilizes a sequence-to-sequence mapping model to infer the latent mapping feature between the input text and the corresponding output suffix generated by the LLM. Then it performs the key token perturbations to obtain more distinguishable membership features. Additionally, considering real-world scenarios where the ground-truth training text samples are limited, a prototype-based membership classifier is introduced to alleviate the overfitting issue. Extensive evaluations on three widely used datasets demonstrate that our framework is effective and superior in the black-box setting. 

---
# Measuring and Guiding Monosemanticity 

**Authors**: Ruben Härle, Felix Friedrich, Manuel Brack, Stephan Wäldchen, Björn Deiseroth, Patrick Schramowski, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2506.19382)  

**Abstract**: There is growing interest in leveraging mechanistic interpretability and controllability to better understand and influence the internal dynamics of large language models (LLMs). However, current methods face fundamental challenges in reliably localizing and manipulating feature representations. Sparse Autoencoders (SAEs) have recently emerged as a promising direction for feature extraction at scale, yet they, too, are limited by incomplete feature isolation and unreliable monosemanticity. To systematically quantify these limitations, we introduce Feature Monosemanticity Score (FMS), a novel metric to quantify feature monosemanticity in latent representation. Building on these insights, we propose Guided Sparse Autoencoders (G-SAE), a method that conditions latent representations on labeled concepts during training. We demonstrate that reliable localization and disentanglement of target concepts within the latent space improve interpretability, detection of behavior, and control. Specifically, our evaluations on toxicity detection, writing style identification, and privacy attribute recognition show that G-SAE not only enhances monosemanticity but also enables more effective and fine-grained steering with less quality degradation. Our findings provide actionable guidelines for measuring and advancing mechanistic interpretability and control of LLMs. 

---
# Spotting Out-of-Character Behavior: Atomic-Level Evaluation of Persona Fidelity in Open-Ended Generation 

**Authors**: Jisu Shin, Juhyun Oh, Eunsu Kim, Hoyun Song, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2506.19352)  

**Abstract**: Ensuring persona fidelity in large language models (LLMs) is essential for maintaining coherent and engaging human-AI interactions. However, LLMs often exhibit Out-of-Character (OOC) behavior, where generated responses deviate from an assigned persona, leading to inconsistencies that affect model reliability. Existing evaluation methods typically assign single scores to entire responses, struggling to capture subtle persona misalignment, particularly in long-form text generation. To address this limitation, we propose an atomic-level evaluation framework that quantifies persona fidelity at a finer granularity. Our three key metrics measure the degree of persona alignment and consistency within and across generations. Our approach enables a more precise and realistic assessment of persona fidelity by identifying subtle deviations that real users would encounter. Through our experiments, we demonstrate that our framework effectively detects persona inconsistencies that prior methods overlook. By analyzing persona fidelity across diverse tasks and personality types, we reveal how task structure and persona desirability influence model adaptability, highlighting challenges in maintaining consistent persona expression. 

---
# JCAPT: A Joint Modeling Approach for CAPT 

**Authors**: Tzu-Hsuan Yang, Yue-Yang He, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19315)  

**Abstract**: Effective pronunciation feedback is critical in second language (L2) learning, for which computer-assisted pronunciation training (CAPT) systems often encompass two key tasks: automatic pronunciation assessment (APA) and mispronunciation detection and diagnosis (MDD). Recent work has shown that joint modeling of these two tasks can yield mutual benefits. Our unified framework leverages Mamba, a selective state space model (SSM), while integrating phonological features and think token strategies to jointly enhance interpretability and fine-grained temporal reasoning in APA and MDD. To our knowledge, this is the first study to combine phonological attribution, SSM-based modeling, and prompting in CAPT. A series of experiments conducted on the speechocean762 benchmark demonstrate that our model consistently outperforms prior methods, particularly on the MDD task. 

---
# EmoStage: A Framework for Accurate Empathetic Response Generation via Perspective-Taking and Phase Recognition 

**Authors**: Zhiyang Qi, Keiko Takamizo, Mariko Ukiyo, Michimasa Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2506.19279)  

**Abstract**: The rising demand for mental health care has fueled interest in AI-driven counseling systems. While large language models (LLMs) offer significant potential, current approaches face challenges, including limited understanding of clients' psychological states and counseling stages, reliance on high-quality training data, and privacy concerns associated with commercial deployment. To address these issues, we propose EmoStage, a framework that enhances empathetic response generation by leveraging the inference capabilities of open-source LLMs without additional training data. Our framework introduces perspective-taking to infer clients' psychological states and support needs, enabling the generation of emotionally resonant responses. In addition, phase recognition is incorporated to ensure alignment with the counseling process and to prevent contextually inappropriate or inopportune responses. Experiments conducted in both Japanese and Chinese counseling settings demonstrate that EmoStage improves the quality of responses generated by base models and performs competitively with data-driven methods. 

---
# What Matters in LLM-generated Data: Diversity and Its Effect on Model Fine-Tuning 

**Authors**: Yuchang Zhu, Zhonghua zhen, Qunshu Lin, Haotong Wei, Xiaolong Sun, Zixuan Yu, Minghao Liu, Zibin Zheng, Liang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19262)  

**Abstract**: With the remarkable generative capabilities of large language models (LLMs), using LLM-generated data to train downstream models has emerged as a promising approach to mitigate data scarcity in specific domains and reduce time-consuming annotations. However, recent studies have highlighted a critical issue: iterative training on self-generated data results in model collapse, where model performance degrades over time. Despite extensive research on the implications of LLM-generated data, these works often neglect the importance of data diversity, a key factor in data quality. In this work, we aim to understand the implications of the diversity of LLM-generated data on downstream model performance. Specifically, we explore how varying levels of diversity in LLM-generated data affect downstream model performance. Additionally, we investigate the performance of models trained on data that mixes different proportions of LLM-generated data, which we refer to as synthetic data. Our experimental results show that, with minimal distribution shift, moderately diverse LLM-generated data can enhance model performance in scenarios with insufficient labeled data, whereas highly diverse generated data has a negative impact. We hope our empirical findings will offer valuable guidance for future studies on LLMs as data generators. 

---
# Personality Prediction from Life Stories using Language Models 

**Authors**: Rasiq Hussain, Jerry Ma, Rithik Khandelwal, Joshua Oltmanns, Mehak Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.19258)  

**Abstract**: Natural Language Processing (NLP) offers new avenues for personality assessment by leveraging rich, open-ended text, moving beyond traditional questionnaires. In this study, we address the challenge of modeling long narrative interview where each exceeds 2000 tokens so as to predict Five-Factor Model (FFM) personality traits. We propose a two-step approach: first, we extract contextual embeddings using sliding-window fine-tuning of pretrained language models; then, we apply Recurrent Neural Networks (RNNs) with attention mechanisms to integrate long-range dependencies and enhance interpretability. This hybrid method effectively bridges the strengths of pretrained transformers and sequence modeling to handle long-context data. Through ablation studies and comparisons with state-of-the-art long-context models such as LLaMA and Longformer, we demonstrate improvements in prediction accuracy, efficiency, and interpretability. Our results highlight the potential of combining language-based features with long-context modeling to advance personality assessment from life narratives. 

---
# Augmenting Multi-Agent Communication with State Delta Trajectory 

**Authors**: Yichen Tang, Weihang Su, Yujia Zhou, Yiqun Liu, Min Zhang, Shaoping Ma, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2506.19209)  

**Abstract**: Multi-agent techniques such as role playing or multi-turn debates have been shown to be effective in improving the performance of large language models (LLMs) in downstream tasks. Despite their differences in workflows, existing LLM-based multi-agent systems mostly use natural language for agent communication. While this is appealing for its simplicity and interpretability, it also introduces inevitable information loss as one model must down sample its continuous state vectors to concrete tokens before transferring them to the other model. Such losses are particularly significant when the information to transfer is not simple facts, but reasoning logics or abstractive thoughts. To tackle this problem, we propose a new communication protocol that transfers both natural language tokens and token-wise state transition trajectory from one agent to another. Particularly, compared to the actual state value, we find that the sequence of state changes in LLMs after generating each token can better reflect the information hidden behind the inference process, so we propose a State Delta Encoding (SDE) method to represent state transition trajectories. The experimental results show that multi-agent systems with SDE achieve SOTA performance compared to other communication protocols, particularly in tasks that involve complex reasoning. This shows the potential of communication augmentation for LLM-based multi-agent systems. 

---
# Prompt, Translate, Fine-Tune, Re-Initialize, or Instruction-Tune? Adapting LLMs for In-Context Learning in Low-Resource Languages 

**Authors**: Christopher Toukmaji, Jeffrey Flanigan  

**Link**: [PDF](https://arxiv.org/pdf/2506.19187)  

**Abstract**: LLMs are typically trained in high-resource languages, and tasks in lower-resourced languages tend to underperform the higher-resource language counterparts for in-context learning. Despite the large body of work on prompting settings, it is still unclear how LLMs should be adapted cross-lingually specifically for in-context learning in the low-resource target languages. We perform a comprehensive study spanning five diverse target languages, three base LLMs, and seven downstream tasks spanning over 4,100 GPU training hours (9,900+ TFLOPs) across various adaptation techniques: few-shot prompting, translate-test, fine-tuning, embedding re-initialization, and instruction fine-tuning. Our results show that the few-shot prompting and translate-test settings tend to heavily outperform the gradient-based adaptation methods. To better understand this discrepancy, we design a novel metric, Valid Output Recall (VOR), and analyze model outputs to empirically attribute the degradation of these trained models to catastrophic forgetting. To the extent of our knowledge, this is the largest study done on in-context learning for low-resource languages with respect to train compute and number of adaptation techniques considered. We make all our datasets and trained models available for public use. 

---
# Enhanced Hybrid Transducer and Attention Encoder Decoder with Text Data 

**Authors**: Yun Tang, Eesung Kim, Vijendra Raj Apsingekar  

**Link**: [PDF](https://arxiv.org/pdf/2506.19159)  

**Abstract**: A joint speech and text optimization method is proposed for hybrid transducer and attention-based encoder decoder (TAED) modeling to leverage large amounts of text corpus and enhance ASR accuracy. The joint TAED (J-TAED) is trained with both speech and text input modalities together, while it only takes speech data as input during inference. The trained model can unify the internal representations from different modalities, and be further extended to text-based domain adaptation. It can effectively alleviate data scarcity for mismatch domain tasks since no speech data is required. Our experiments show J-TAED successfully integrates speech and linguistic information into one model, and reduce the WER by 5.8 ~12.8% on the Librispeech dataset. The model is also evaluated on two out-of-domain datasets: one is finance and another is named entity focused. The text-based domain adaptation brings 15.3% and 17.8% WER reduction on those two datasets respectively. 

---
# Human-Aligned Faithfulness in Toxicity Explanations of LLMs 

**Authors**: Ramaravind K. Mothilal, Joanna Roy, Syed Ishtiaque Ahmed, Shion Guha  

**Link**: [PDF](https://arxiv.org/pdf/2506.19113)  

**Abstract**: The discourse around toxicity and LLMs in NLP largely revolves around detection tasks. This work shifts the focus to evaluating LLMs' reasoning about toxicity -- from their explanations that justify a stance -- to enhance their trustworthiness in downstream tasks. Despite extensive research on explainability, it is not straightforward to adopt existing methods to evaluate free-form toxicity explanation due to their over-reliance on input text perturbations, among other challenges. To account for these, we propose a novel, theoretically-grounded multi-dimensional criterion, Human-Aligned Faithfulness (HAF), that measures the extent to which LLMs' free-form toxicity explanations align with those of a rational human under ideal conditions. We develop six metrics, based on uncertainty quantification, to comprehensively evaluate \haf of LLMs' toxicity explanations with no human involvement, and highlight how "non-ideal" the explanations are. We conduct several experiments on three Llama models (of size up to 70B) and an 8B Ministral model on five diverse toxicity datasets. Our results show that while LLMs generate plausible explanations to simple prompts, their reasoning about toxicity breaks down when prompted about the nuanced relations between the complete set of reasons, the individual reasons, and their toxicity stances, resulting in inconsistent and nonsensical responses. We open-source our code and LLM-generated explanations at this https URL. 

---
# Language Models Might Not Understand You: Evaluating Theory of Mind via Story Prompting 

**Authors**: Nathaniel Getachew, Abulhair Saparov  

**Link**: [PDF](https://arxiv.org/pdf/2506.19089)  

**Abstract**: We introduce $\texttt{StorySim}$, a programmable framework for synthetically generating stories to evaluate the theory of mind (ToM) and world modeling (WM) capabilities of large language models (LLMs). Unlike prior benchmarks that may suffer from contamination in pretraining data, $\texttt{StorySim}$ produces novel, compositional story prompts anchored by a highly controllable $\texttt{Storyboard}$, enabling precise manipulation of character perspectives and events. We use this framework to design first- and second-order ToM tasks alongside WM tasks that control for the ability to track and model mental states. Our experiments across a suite of state-of-the-art LLMs reveal that most models perform better on WM tasks than ToM tasks, and that models tend to perform better reasoning with humans compared to inanimate objects. Additionally, our framework enabled us to find evidence of heuristic behavior such as recency bias and an over-reliance on earlier events in the story. All code for generating data and evaluations is freely available. 

---
# MFTCXplain: A Multilingual Benchmark Dataset for Evaluating the Moral Reasoning of LLMs through Hate Speech Multi-hop Explanation 

**Authors**: Jackson Trager, Francielle Vargas, Diego Alves, Matteo Guida, Mikel K. Ngueajio, Ameeta Agrawal, Flor Plaza-del-Arco, Yalda Daryanai, Farzan Karimi-Malekabadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.19073)  

**Abstract**: Ensuring the moral reasoning capabilities of Large Language Models (LLMs) is a growing concern as these systems are used in socially sensitive tasks. Nevertheless, current evaluation benchmarks present two major shortcomings: a lack of annotations that justify moral classifications, which limits transparency and interpretability; and a predominant focus on English, which constrains the assessment of moral reasoning across diverse cultural settings. In this paper, we introduce MFTCXplain, a multilingual benchmark dataset for evaluating the moral reasoning of LLMs via hate speech multi-hop explanation using Moral Foundation Theory (MFT). The dataset comprises 3,000 tweets across Portuguese, Italian, Persian, and English, annotated with binary hate speech labels, moral categories, and text span-level rationales. Empirical results highlight a misalignment between LLM outputs and human annotations in moral reasoning tasks. While LLMs perform well in hate speech detection (F1 up to 0.836), their ability to predict moral sentiments is notably weak (F1 < 0.35). Furthermore, rationale alignment remains limited mainly in underrepresented languages. These findings show the limited capacity of current LLMs to internalize and reflect human moral reasoning. 

---
# NLPnorth @ TalentCLEF 2025: Comparing Discriminative, Contrastive, and Prompt-Based Methods for Job Title and Skill Matching 

**Authors**: Mike Zhang, Rob van der Goot  

**Link**: [PDF](https://arxiv.org/pdf/2506.19058)  

**Abstract**: Matching job titles is a highly relevant task in the computational job market domain, as it improves e.g., automatic candidate matching, career path prediction, and job market analysis. Furthermore, aligning job titles to job skills can be considered an extension to this task, with similar relevance for the same downstream tasks. In this report, we outline NLPnorth's submission to TalentCLEF 2025, which includes both of these tasks: Multilingual Job Title Matching, and Job Title-Based Skill Prediction. For both tasks we compare (fine-tuned) classification-based, (fine-tuned) contrastive-based, and prompting methods. We observe that for Task A, our prompting approach performs best with an average of 0.492 mean average precision (MAP) on test data, averaged over English, Spanish, and German. For Task B, we obtain an MAP of 0.290 on test data with our fine-tuned classification-based approach. Additionally, we made use of extra data by pulling all the language-specific titles and corresponding \emph{descriptions} from ESCO for each job and skill. Overall, we find that the largest multilingual language models perform best for both tasks. Per the provisional results and only counting the unique teams, the ranking on Task A is 5$^{\text{th}}$/20 and for Task B 3$^{\text{rd}}$/14. 

---
# Plan for Speed -- Dilated Scheduling for Masked Diffusion Language Models 

**Authors**: Omer Luxembourg, Haim Permuter, Eliya Nachmani  

**Link**: [PDF](https://arxiv.org/pdf/2506.19037)  

**Abstract**: Masked diffusion language models (MDLM) have shown strong promise for non-autoregressive text generation, yet existing samplers act as implicit planners, selecting tokens to unmask via denoiser confidence or entropy scores. Such heuristics falter under parallel unmasking - they ignore pairwise interactions between tokens and cannot account for dependencies when unmasking multiple positions at once, limiting their inference time to traditional auto-regressive (AR) models. We introduce the Dilated-scheduled Unmasking Strategy (DUS), an inference-only, planner-model-free method that requires no additional training. DUS leverages a first-order Markov assumption to partition sequence positions into dilation-based groups of non-adjacent tokens, enabling independent, parallel unmasking steps that respect local context that minimizes the joint entropy of each iteration step. Unlike semi-AR block approaches (e.g., LLADA and Dream) that still invoke the denoiser per block, DUS reduces the number of denoiser calls to O(log B) per generation block - yielding substantial speedup over the O(B) run time of state-of-the-art diffusion models, where B is the block size in the semi-AR inference process. In experiments on math (GSM8K) and code completion (Humaneval, MBPP) benchmarks - domains suited to non-ordinal generation - DUS improves scores over parallel confidence-based planner, without modifying the underlying denoiser. DUS offers a lightweight, budget-aware approach to efficient, high-quality text generation, paving the way to unlock the true capabilities of MDLMs. 

---
# Quantifying Fairness in LLMs Beyond Tokens: A Semantic and Statistical Perspective 

**Authors**: Weijie Xu, Yiwen Wang, Chi Xue, Xiangkun Hu, Xi Fang, Guimin Dong, Chandan K. Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2506.19028)  

**Abstract**: Large Language Models (LLMs) often generate responses with inherent biases, undermining their reliability in real-world applications. Existing evaluation methods often overlook biases in long-form responses and the intrinsic variability of LLM outputs. To address these challenges, we propose FiSCo(Fine-grained Semantic Computation), a novel statistical framework to evaluate group-level fairness in LLMs by detecting subtle semantic differences in long-form responses across demographic groups. Unlike prior work focusing on sentiment or token-level comparisons, FiSCo goes beyond surface-level analysis by operating at the claim level, leveraging entailment checks to assess the consistency of meaning across responses. We decompose model outputs into semantically distinct claims and apply statistical hypothesis testing to compare inter- and intra-group similarities, enabling robust detection of subtle biases. We formalize a new group counterfactual fairness definition and validate FiSCo on both synthetic and human-annotated datasets spanning gender, race, and age. Experiments show that FiSco more reliably identifies nuanced biases while reducing the impact of stochastic LLM variability, outperforming various evaluation metrics. 

---
# Broken Tokens? Your Language Model can Secretly Handle Non-Canonical Tokenizations 

**Authors**: Brian Siyuan Zheng, Alisa Liu, Orevaoghene Ahia, Jonathan Hayase, Yejin Choi, Noah A. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2506.19004)  

**Abstract**: Modern tokenizers employ deterministic algorithms to map text into a single "canonical" token sequence, yet the same string can be encoded as many non-canonical tokenizations using the tokenizer vocabulary. In this work, we investigate the robustness of LMs to text encoded with non-canonical tokenizations entirely unseen during training. Surprisingly, when evaluated across 20 benchmarks, we find that instruction-tuned models retain up to 93.4% of their original performance when given a randomly sampled tokenization, and 90.8% with character-level tokenization. We see that overall stronger models tend to be more robust, and robustness diminishes as the tokenization departs farther from the canonical form. Motivated by these results, we then identify settings where non-canonical tokenization schemes can *improve* performance, finding that character-level segmentation improves string manipulation and code understanding tasks by up to +14%, and right-aligned digit grouping enhances large-number arithmetic by +33%. Finally, we investigate the source of this robustness, finding that it arises in the instruction-tuning phase. We show that while both base and post-trained models grasp the semantics of non-canonical tokenizations (perceiving them as containing misspellings), base models try to mimic the imagined mistakes and degenerate into nonsensical output, while post-trained models are committed to fluent responses. Overall, our findings suggest that models are less tied to their tokenizer than previously believed, and demonstrate the promise of intervening on tokenization at inference time to boost performance. 

---
# Mirage of Mastery: Memorization Tricks LLMs into Artificially Inflated Self-Knowledge 

**Authors**: Sahil Kale, Vijaykant Nadadur  

**Link**: [PDF](https://arxiv.org/pdf/2506.18998)  

**Abstract**: When artificial intelligence mistakes memorization for intelligence, it creates a dangerous mirage of reasoning. Existing studies treat memorization and self-knowledge deficits in LLMs as separate issues and do not recognize an intertwining link that degrades the trustworthiness of LLM responses. In our study, we utilize a novel framework to ascertain if LLMs genuinely learn reasoning patterns from training data or merely memorize them to assume competence across problems of similar complexity focused on STEM domains. Our analysis shows a noteworthy problem in generalization: LLMs draw confidence from memorized solutions to infer a higher self-knowledge about their reasoning ability, which manifests as an over 45% inconsistency in feasibility assessments when faced with self-validated, logically coherent task perturbations. This effect is most pronounced in science and medicine domains, which tend to have maximal standardized jargon and problems, further confirming our approach. Significant wavering within the self-knowledge of LLMs also shows flaws in current architectures and training patterns, highlighting the need for techniques that ensure a balanced, consistent stance on models' perceptions of their own knowledge for maximum AI explainability and trustworthiness. Our code and results are available publicly at this https URL. 

---
# MemeMind: A Large-Scale Multimodal Dataset with Chain-of-Thought Reasoning for Harmful Meme Detection 

**Authors**: Hexiang Gu, Qifan Yu, Saihui Hou, Zhiqin Fang, Huijia Wu, Zhaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2506.18919)  

**Abstract**: The rapid development of social media has intensified the spread of harmful content. Harmful memes, which integrate both images and text, pose significant challenges for automated detection due to their implicit semantics and complex multimodal interactions. Although existing research has made progress in detection accuracy and interpretability, the lack of a systematic, large-scale, diverse, and highly explainable dataset continues to hinder further advancement in this field. To address this gap, we introduce MemeMind, a novel dataset featuring scientifically rigorous standards, large scale, diversity, bilingual support (Chinese and English), and detailed Chain-of-Thought (CoT) annotations. MemeMind fills critical gaps in current datasets by offering comprehensive labeling and explicit reasoning traces, thereby providing a solid foundation for enhancing harmful meme detection. In addition, we propose an innovative detection framework, MemeGuard, which effectively integrates multimodal information with reasoning process modeling, significantly improving models' ability to understand and identify harmful memes. Extensive experiments conducted on the MemeMind dataset demonstrate that MemeGuard consistently outperforms existing state-of-the-art methods in harmful meme detection tasks. 

---
# ScaleCap: Inference-Time Scalable Image Captioning via Dual-Modality Debiasing 

**Authors**: Long Xing, Qidong Huang, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Jinsong Li, Shuangrui Ding, Weiming Zhang, Nenghai Yu, Jiaqi Wang, Feng Wu, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.19848)  

**Abstract**: This paper presents ScaleCap, an inference-time scalable image captioning strategy that generates comprehensive and detailed image captions. The key challenges of high-quality image captioning lie in the inherent biases of LVLMs: multimodal bias resulting in imbalanced descriptive granularity, offering detailed accounts of some elements while merely skimming over others; linguistic bias leading to hallucinated descriptions of non-existent objects. To address these issues, we propose a scalable debiased captioning strategy, which continuously enriches and calibrates the caption with increased inference budget. Specifically, we propose two novel components: heuristic question answering and contrastive sentence rating. The former generates content-specific questions based on the image and answers them to progressively inject relevant information into the caption. The latter employs sentence-level offline contrastive decoding to effectively identify and eliminate hallucinations caused by linguistic biases. With increased inference cost, more heuristic questions are raised by ScaleCap to progressively capture additional visual details, generating captions that are more accurate, balanced, and informative. Extensive modality alignment experiments demonstrate the effectiveness of ScaleCap. Annotating 450K images with ScaleCap and using them for LVLM pretraining leads to consistent performance gains across 11 widely used benchmarks. Furthermore, ScaleCap showcases superb richness and fidelity of generated captions with two additional tasks: replacing images with captions in VQA task, and reconstructing images from captions to assess semantic coverage. Code is available at this https URL. 

---
# Orthogonal Finetuning Made Scalable 

**Authors**: Zeju Qiu, Weiyang Liu, Adrian Weller, Bernhard Schölkopf  

**Link**: [PDF](https://arxiv.org/pdf/2506.19847)  

**Abstract**: Orthogonal finetuning (OFT) offers highly parameter-efficient adaptation while preventing catastrophic forgetting, but its high runtime and memory demands limit practical deployment. We identify the core computational bottleneck in OFT as its weight-centric implementation, which relies on costly matrix-matrix multiplications with cubic complexity. To overcome this, we propose OFTv2, an input-centric reformulation that instead uses matrix-vector multiplications (i.e., matrix-free computation), reducing the computational cost to quadratic. We further introduce the Cayley-Neumann parameterization, an efficient orthogonal parameterization that approximates the matrix inversion in Cayley transform via a truncated Neumann series. These modifications allow OFTv2 to achieve up to 10x faster training and 3x lower GPU memory usage without compromising performance. In addition, we extend OFTv2 to support finetuning quantized foundation models and show that it outperforms the popular QLoRA in training stability, efficiency, and memory usage. 

---
# Scaling Speculative Decoding with Lookahead Reasoning 

**Authors**: Yichao Fu, Rui Ge, Zelei Shao, Zhijie Deng, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19830)  

**Abstract**: Reasoning models excel by generating long chain-of-thoughts, but decoding the resulting thousands of tokens is slow. Token-level speculative decoding (SD) helps, but its benefit is capped, because the chance that an entire $\gamma$-token guess is correct falls exponentially as $\gamma$ grows. This means allocating more compute for longer token drafts faces an algorithmic ceiling -- making the speedup modest and hardware-agnostic. We raise this ceiling with Lookahead Reasoning, which exploits a second, step-level layer of parallelism. Our key insight is that reasoning models generate step-by-step, and each step needs only to be semantically correct, not exact token matching. In Lookahead Reasoning, a lightweight draft model proposes several future steps; the target model expands each proposal in one batched pass, and a verifier keeps semantically correct steps while letting the target regenerate any that fail. Token-level SD still operates within each reasoning step, so the two layers of parallelism multiply. We show Lookahead Reasoning lifts the peak speedup of SD both theoretically and empirically. Across GSM8K, AIME, and other benchmarks, Lookahead Reasoning improves the speedup of SD from 1.4x to 2.1x while preserving answer quality, and its speedup scales better with additional GPU throughput. Our code is available at this https URL 

---
# Evaluating Compliance with Visualization Guidelines in Diagrams for Scientific Publications Using Large Vision Language Models 

**Authors**: Johannes Rückert, Louise Bloch, Christoph M. Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2506.19825)  

**Abstract**: Diagrams are widely used to visualize data in publications. The research field of data visualization deals with defining principles and guidelines for the creation and use of these diagrams, which are often not known or adhered to by researchers, leading to misinformation caused by providing inaccurate or incomplete information.
In this work, large Vision Language Models (VLMs) are used to analyze diagrams in order to identify potential problems in regards to selected data visualization principles and guidelines. To determine the suitability of VLMs for these tasks, five open source VLMs and five prompting strategies are compared using a set of questions derived from selected data visualization guidelines.
The results show that the employed VLMs work well to accurately analyze diagram types (F1-score 82.49 %), 3D effects (F1-score 98.55 %), axes labels (F1-score 76.74 %), lines (RMSE 1.16), colors (RMSE 1.60) and legends (F1-score 96.64 %, RMSE 0.70), while they cannot reliably provide feedback about the image quality (F1-score 0.74 %) and tick marks/labels (F1-score 46.13 %). Among the employed VLMs, Qwen2.5VL performs best, and the summarizing prompting strategy performs best for most of the experimental questions.
It is shown that VLMs can be used to automatically identify a number of potential issues in diagrams, such as missing axes labels, missing legends, and unnecessary 3D effects. The approach laid out in this work can be extended for further aspects of data visualization. 

---
# KnowRL: Exploring Knowledgeable Reinforcement Learning for Factuality 

**Authors**: Baochang Ren, Shuofei Qiao, Wenhao Yu, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19807)  

**Abstract**: Large Language Models (LLMs), particularly slow-thinking models, often exhibit severe hallucination, outputting incorrect content due to an inability to accurately recognize knowledge boundaries during reasoning. While Reinforcement Learning (RL) can enhance complex reasoning abilities, its outcome-oriented reward mechanism often lacks factual supervision over the thinking process, further exacerbating the hallucination problem. To address the high hallucination in slow-thinking models, we propose Knowledge-enhanced RL, KnowRL. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. This targeted factual input during RL training enables the model to learn and internalize fact-based reasoning strategies. By directly rewarding adherence to facts within the reasoning steps, KnowRL fosters a more reliable thinking process. Experimental results on three hallucination evaluation datasets and two reasoning evaluation datasets demonstrate that KnowRL effectively mitigates hallucinations in slow-thinking models while maintaining their original strong reasoning capabilities. Our code is available at this https URL. 

---
# LLM-Based Social Simulations Require a Boundary 

**Authors**: Zengqing Wu, Run Peng, Takayuki Ito, Chuan Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.19806)  

**Abstract**: This position paper argues that large language model (LLM)-based social simulations should establish clear boundaries to meaningfully contribute to social science research. While LLMs offer promising capabilities for modeling human-like agents compared to traditional agent-based modeling, they face fundamental limitations that constrain their reliability for social pattern discovery. The core issue lies in LLMs' tendency towards an ``average persona'' that lacks sufficient behavioral heterogeneity, a critical requirement for simulating complex social dynamics. We examine three key boundary problems: alignment (simulated behaviors matching real-world patterns), consistency (maintaining coherent agent behavior over time), and robustness (reproducibility under varying conditions). We propose heuristic boundaries for determining when LLM-based simulations can reliably advance social science understanding. We believe that these simulations are more valuable when focusing on (1) collective patterns rather than individual trajectories, (2) agent behaviors aligning with real population averages despite limited variance, and (3) proper validation methods available for testing simulation robustness. We provide a practical checklist to guide researchers in determining the appropriate scope and claims for LLM-based social simulations. 

---
# Kling-Foley: Multimodal Diffusion Transformer for High-Quality Video-to-Audio Generation 

**Authors**: Jun Wang, Xijuan Zeng, Chunyu Qiang, Ruilong Chen, Shiyao Wang, Le Wang, Wangjing Zhou, Pengfei Cai, Jiahui Zhao, Nan Li, Zihan Li, Yuzhe Liang, Xiaopeng Wang, Haorui Zheng, Ming Wen, Kang Yin, Yiran Wang, Nan Li, Feng Deng, Liang Dong, Chen Zhang, Di Zhang, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2506.19774)  

**Abstract**: We propose Kling-Foley, a large-scale multimodal Video-to-Audio generation model that synthesizes high-quality audio synchronized with video content. In Kling-Foley, we introduce multimodal diffusion transformers to model the interactions between video, audio, and text modalities, and combine it with a visual semantic representation module and an audio-visual synchronization module to enhance alignment capabilities. Specifically, these modules align video conditions with latent audio elements at the frame level, thereby improving semantic alignment and audio-visual synchronization. Together with text conditions, this integrated approach enables precise generation of video-matching sound effects. In addition, we propose a universal latent audio codec that can achieve high-quality modeling in various scenarios such as sound effects, speech, singing, and music. We employ a stereo rendering method that imbues synthesized audio with a spatial presence. At the same time, in order to make up for the incomplete types and annotations of the open-source benchmark, we also open-source an industrial-level benchmark Kling-Audio-Eval. Our experiments show that Kling-Foley trained with the flow matching objective achieves new audio-visual SOTA performance among public models in terms of distribution matching, semantic alignment, temporal alignment and audio quality. 

---
# NEAR$^2$: A Nested Embedding Approach to Efficient Product Retrieval and Ranking 

**Authors**: Shenbin Qian, Diptesh Kanojia, Samarth Agrawal, Hadeel Saadany, Swapnil Bhosale, Constantin Orasan, Zhe Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19743)  

**Abstract**: E-commerce information retrieval (IR) systems struggle to simultaneously achieve high accuracy in interpreting complex user queries and maintain efficient processing of vast product catalogs. The dual challenge lies in precisely matching user intent with relevant products while managing the computational demands of real-time search across massive inventories. In this paper, we propose a Nested Embedding Approach to product Retrieval and Ranking, called NEAR$^2$, which can achieve up to $12$ times efficiency in embedding size at inference time while introducing no extra cost in training and improving performance in accuracy for various encoder-based Transformer models. We validate our approach using different loss functions for the retrieval and ranking task, including multiple negative ranking loss and online contrastive loss, on four different test sets with various IR challenges such as short and implicit queries. Our approach achieves an improved performance over a smaller embedding dimension, compared to any existing models. 

---
# Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models 

**Authors**: Jungwoo Park, Taewhoo Lee, Chanwoong Yoon, Hyeon Hwang, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19697)  

**Abstract**: Extreme activation outliers in Large Language Models (LLMs) critically degrade quantization performance, hindering efficient on-device deployment. While channel-wise operations and adaptive gradient scaling are recognized causes, practical mitigation remains challenging. We introduce Outlier-Safe Pre-Training (OSP), a practical guideline that proactively prevents outlier formation rather than relying on post-hoc mitigation. OSP combines three key innovations: (1) the Muon optimizer, eliminating privileged bases while maintaining training efficiency; (2) Single-Scale RMSNorm, preventing channel-wise amplification; and (3) a learnable embedding projection, redistributing activation magnitudes originating from embedding matrices. We validate OSP by training a 1.4B-parameter model on 1 trillion tokens, which is the first production-scale LLM trained without such outliers. Under aggressive 4-bit quantization, our OSP model achieves a 35.7 average score across 10 benchmarks (compared to 26.5 for an Adam-trained model), with only a 2% training overhead. Remarkably, OSP models exhibit near-zero excess kurtosis (0.04) compared to extreme values (1818.56) in standard models, fundamentally altering LLM quantization behavior. Our work demonstrates that outliers are not inherent to LLMs but are consequences of training strategies, paving the way for more efficient LLM deployment. The source code and pretrained checkpoints are available at this https URL. 

---
# Recurrent Visual Feature Extraction and Stereo Attentions for CT Report Generation 

**Authors**: Yuanhe Tian, Lei Mao, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.19665)  

**Abstract**: Generating reports for computed tomography (CT) images is a challenging task, while similar to existing studies for medical image report generation, yet has its unique characteristics, such as spatial encoding of multiple images, alignment between image volume and texts, etc. Existing solutions typically use general 2D or 3D image processing techniques to extract features from a CT volume, where they firstly compress the volume and then divide the compressed CT slices into patches for visual encoding. These approaches do not explicitly account for the transformations among CT slices, nor do they effectively integrate multi-level image features, particularly those containing specific organ lesions, to instruct CT report generation (CTRG). In considering the strong correlation among consecutive slices in CT scans, in this paper, we propose a large language model (LLM) based CTRG method with recurrent visual feature extraction and stereo attentions for hierarchical feature modeling. Specifically, we use a vision Transformer to recurrently process each slice in a CT volume, and employ a set of attentions over the encoded slices from different perspectives to selectively obtain important visual information and align them with textual features, so as to better instruct an LLM for CTRG. Experiment results and further analysis on the benchmark M3D-Cap dataset show that our method outperforms strong baseline models and achieves state-of-the-art results, demonstrating its validity and effectiveness. 

---
# Fake or Real, Can Robots Tell? Evaluating Embodied Vision-Language Models on Real and 3D-Printed Objects 

**Authors**: Federico Tavella, Kathryn Mearns, Angelo Cangelosi  

**Link**: [PDF](https://arxiv.org/pdf/2506.19579)  

**Abstract**: Robotic scene understanding increasingly relies on vision-language models (VLMs) to generate natural language descriptions of the environment. In this work, we present a comparative study of captioning strategies for tabletop scenes captured by a robotic arm equipped with an RGB camera. The robot collects images of objects from multiple viewpoints, and we evaluate several models that generate scene descriptions. We compare the performance of various captioning models, like BLIP and VLMs. Our experiments examine the trade-offs between single-view and multi-view captioning, and difference between recognising real-world and 3D printed objects. We quantitatively evaluate object identification accuracy, completeness, and naturalness of the generated captions. Results show that VLMs can be used in robotic settings where common objects need to be recognised, but fail to generalise to novel representations. Our findings provide practical insights into deploying foundation models for embodied agents in real-world settings. 

---
# NaviAgent: Bilevel Planning on Tool Dependency Graphs for Function Calling 

**Authors**: Yan Jiang, Hao Zhou, LiZhong GU, Ai Han, TianLong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.19500)  

**Abstract**: LLMs' reliance on static knowledge and fragile tool invocation severely hinders the orchestration of complex, heterogeneous toolchains, particularly at large scales. Existing methods typically use rigid single-path execution, resulting in poor error recovery and exponentially growing search spaces. We introduce NaviAgent, a graph-navigated bilevel planning architecture for robust function calling, comprising a Multi-Path Decider and Graph-Encoded Navigator. As an LLM-powered agent, the Multi-Path Decider defines a four-dimensional decision space and continuously perceives environmental states, dynamically selecting the optimal action to fully cover all tool invocation scenarios. The Graph-Encoded Navigator constructs a Tool Dependency Heterogeneous Graph (TDHG), where node embeddings explicitly fuse API schema structure with historical invocation behavior. It also integrates a novel heuristic search strategy that guides the Decider toward efficient and highly successful toolchains, even for unseen tool combinations. Experiments show that NaviAgent consistently achieves the highest task success rate (TSR) across all foundation models and task complexities, outperforming the average baselines (ReAct, ToolLLM, {\alpha}-UMI) by 13.5%, 16.4%, and 19.0% on Qwen2.5-14B, Qwen2.5-32B, and Deepseek-V3, respectively. Its execution steps are typically within one step of the most efficient baseline, ensuring a strong balance between quality and efficiency. Notably, a fine-tuned Qwen2.5-14B model achieves a TSR of 49.5%, surpassing the much larger 32B model (44.9%) under our architecture. Incorporating the Graph-Encoded Navigator further boosts TSR by an average of 2.4 points, with gains up over 9 points on complex tasks for larger models (Deepseek-V3 and GPT-4o), highlighting its essential role in toolchain orchestration. 

---
# TTSDS2: Resources and Benchmark for Evaluating Human-Quality Text to Speech Systems 

**Authors**: Christoph Minixhofer, Ondrej Klejch, Peter Bell  

**Link**: [PDF](https://arxiv.org/pdf/2506.19441)  

**Abstract**: Evaluation of Text to Speech (TTS) systems is challenging and resource-intensive. Subjective metrics such as Mean Opinion Score (MOS) are not easily comparable between works. Objective metrics are frequently used, but rarely validated against subjective ones. Both kinds of metrics are challenged by recent TTS systems capable of producing synthetic speech indistinguishable from real speech. In this work, we introduce Text to Speech Distribution Score 2 (TTSDS2), a more robust and improved version of TTSDS. Across a range of domains and languages, it is the only one out of 16 compared metrics to correlate with a Spearman correlation above 0.50 for every domain and subjective score evaluated. We also release a range of resources for evaluating synthetic speech close to real speech: A dataset with over 11,000 subjective opinion score ratings; a pipeline for continually recreating a multilingual test dataset to avoid data leakage; and a continually updated benchmark for TTS in 14 languages. 

---
# Mem4Nav: Boosting Vision-and-Language Navigation in Urban Environments with a Hierarchical Spatial-Cognition Long-Short Memory System 

**Authors**: Lixuan He, Haoyu Dong, Zhenxing Chen, Yangcheng Yu, Jie Feng, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.19433)  

**Abstract**: Vision-and-Language Navigation (VLN) in large-scale urban environments requires embodied agents to ground linguistic instructions in complex scenes and recall relevant experiences over extended time horizons. Prior modular pipelines offer interpretability but lack unified memory, while end-to-end (M)LLM agents excel at fusing vision and language yet remain constrained by fixed context windows and implicit spatial reasoning. We introduce \textbf{Mem4Nav}, a hierarchical spatial-cognition long-short memory system that can augment any VLN backbone. Mem4Nav fuses a sparse octree for fine-grained voxel indexing with a semantic topology graph for high-level landmark connectivity, storing both in trainable memory tokens embedded via a reversible Transformer. Long-term memory (LTM) compresses and retains historical observations at both octree and graph nodes, while short-term memory (STM) caches recent multimodal entries in relative coordinates for real-time obstacle avoidance and local planning. At each step, STM retrieval sharply prunes dynamic context, and, when deeper history is needed, LTM tokens are decoded losslessly to reconstruct past embeddings. Evaluated on Touchdown and Map2Seq across three backbones (modular, state-of-the-art VLN with prompt-based LLM, and state-of-the-art VLN with strided-attention MLLM), Mem4Nav yields 7-13 pp gains in Task Completion, sufficient SPD reduction, and >10 pp nDTW improvement. Ablations confirm the indispensability of both the hierarchical map and dual memory modules. Our codes are open-sourced via this https URL. 

---
# In-Context Occam's Razor: How Transformers Prefer Simpler Hypotheses on the Fly 

**Authors**: Puneesh Deora, Bhavya Vasudeva, Tina Behnia, Christos Thrampoulidis  

**Link**: [PDF](https://arxiv.org/pdf/2506.19351)  

**Abstract**: In-context learning (ICL) enables transformers to adapt to new tasks through contextual examples without parameter updates. While existing research has typically studied ICL in fixed-complexity environments, practical language models encounter tasks spanning diverse complexity levels. This paper investigates how transformers navigate hierarchical task structures where higher-complexity categories can perfectly represent any pattern generated by simpler ones. We design well-controlled testbeds based on Markov chains and linear regression that reveal transformers not only identify the appropriate complexity level for each task but also accurately infer the corresponding parameters--even when the in-context examples are compatible with multiple complexity hypotheses. Notably, when presented with data generated by simpler processes, transformers consistently favor the least complex sufficient explanation. We theoretically explain this behavior through a Bayesian framework, demonstrating that transformers effectively implement an in-context Bayesian Occam's razor by balancing model fit against complexity penalties. We further ablate on the roles of model size, training mixture distribution, inference context length, and architecture. Finally, we validate this Occam's razor-like inductive bias on a pretrained GPT-4 model with Boolean-function tasks as case study, suggesting it may be inherent to transformers trained on diverse task distributions. 

---
# Skywork-SWE: Unveiling Data Scaling Laws for Software Engineering in LLMs 

**Authors**: Liang Zeng, Yongcong Li, Yuzhen Xiao, Changshi Li, Chris Yuhao Liu, Rui Yan, Tianwen Wei, Jujie He, Xuchen Song, Yang Liu, Yahui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.19290)  

**Abstract**: Software engineering (SWE) has recently emerged as a crucial testbed for next-generation LLM agents, demanding inherent capabilities in two critical dimensions: sustained iterative problem-solving (e.g., >50 interaction rounds) and long-context dependency resolution (e.g., >32k tokens). However, the data curation process in SWE remains notoriously time-consuming, as it heavily relies on manual annotation for code file filtering and the setup of dedicated runtime environments to execute and validate unit tests. Consequently, most existing datasets are limited to only a few thousand GitHub-sourced instances. To this end, we propose an incremental, automated data-curation pipeline that systematically scales both the volume and diversity of SWE datasets. Our dataset comprises 10,169 real-world Python task instances from 2,531 distinct GitHub repositories, each accompanied by a task specified in natural language and a dedicated runtime-environment image for automated unit-test validation. We have carefully curated over 8,000 successfully runtime-validated training trajectories from our proposed SWE dataset. When fine-tuning the Skywork-SWE model on these trajectories, we uncover a striking data scaling phenomenon: the trained model's performance for software engineering capabilities in LLMs continues to improve as the data size increases, showing no signs of saturation. Notably, our Skywork-SWE model achieves 38.0% pass@1 accuracy on the SWE-bench Verified benchmark without using verifiers or multiple rollouts, establishing a new state-of-the-art (SOTA) among the Qwen2.5-Coder-32B-based LLMs built on the OpenHands agent framework. Furthermore, with the incorporation of test-time scaling techniques, the performance further improves to 47.0% accuracy, surpassing the previous SOTA results for sub-32B parameter models. We release the Skywork-SWE-32B model checkpoint to accelerate future research. 

---
# Bayesian Evolutionary Swarm Architecture: A Formal Epistemic System Grounded in Truth-Based Competition 

**Authors**: Craig Steven Wright  

**Link**: [PDF](https://arxiv.org/pdf/2506.19191)  

**Abstract**: We introduce a mathematically rigorous framework for an artificial intelligence system composed of probabilistic agents evolving through structured competition and belief revision. The architecture, grounded in Bayesian inference, measure theory, and population dynamics, defines agent fitness as a function of alignment with a fixed external oracle representing ground truth. Agents compete in a discrete-time environment, adjusting posterior beliefs through observed outcomes, with higher-rated agents reproducing and lower-rated agents undergoing extinction. Ratings are updated via pairwise truth-aligned utility comparisons, and belief updates preserve measurable consistency and stochastic convergence. We introduce hash-based cryptographic identity commitments to ensure traceability, alongside causal inference operators using do-calculus. Formal theorems on convergence, robustness, and evolutionary stability are provided. The system establishes truth as an evolutionary attractor, demonstrating that verifiable knowledge arises from adversarial epistemic pressure within a computable, self-regulating swarm. 

---
# Thought Anchors: Which LLM Reasoning Steps Matter? 

**Authors**: Paul C. Bogdan, Uzay Macar, Neel Nanda, Arthur Conmy  

**Link**: [PDF](https://arxiv.org/pdf/2506.19143)  

**Abstract**: Reasoning large language models have recently achieved state-of-the-art performance in many fields. However, their long-form chain-of-thought reasoning creates interpretability challenges as each generated token depends on all previous ones, making the computation harder to decompose. We argue that analyzing reasoning traces at the sentence level is a promising approach to understanding reasoning processes. We present three complementary attribution methods: (1) a black-box method measuring each sentence's counterfactual importance by comparing final answers across 100 rollouts conditioned on the model generating that sentence or one with a different meaning; (2) a white-box method of aggregating attention patterns between pairs of sentences, which identified ``broadcasting'' sentences that receive disproportionate attention from all future sentences via ``receiver'' attention heads; (3) a causal attribution method measuring logical connections between sentences by suppressing attention toward one sentence and measuring the effect on each future sentence's tokens. Each method provides evidence for the existence of thought anchors, reasoning steps that have outsized importance and that disproportionately influence the subsequent reasoning process. These thought anchors are typically planning or backtracking sentences. We provide an open-source tool (this http URL) for visualizing the outputs of our methods, and present a case study showing converging patterns across methods that map how a model performs multi-step reasoning. The consistency across methods demonstrates the potential of sentence-level analysis for a deeper understanding of reasoning models. 

---
# HAWAII: Hierarchical Visual Knowledge Transfer for Efficient Vision-Language Models 

**Authors**: Yimu Wang, Mozhgan Nasr Azadani, Sean Sedwards, Krzysztof Czarnecki  

**Link**: [PDF](https://arxiv.org/pdf/2506.19072)  

**Abstract**: Improving the visual understanding ability of vision-language models (VLMs) is crucial for enhancing their performance across various tasks. While using multiple pretrained visual experts has shown great promise, it often incurs significant computational costs during training and inference. To address this challenge, we propose HAWAII, a novel framework that distills knowledge from multiple visual experts into a single vision encoder, enabling it to inherit the complementary strengths of several experts with minimal computational overhead. To mitigate conflicts among different teachers and switch between different teacher-specific knowledge, instead of using a fixed set of adapters for multiple teachers, we propose to use teacher-specific Low-Rank Adaptation (LoRA) adapters with a corresponding router. Each adapter is aligned with a specific teacher, avoiding noisy guidance during distillation. To enable efficient knowledge distillation, we propose fine-grained and coarse-grained distillation. At the fine-grained level, token importance scores are employed to emphasize the most informative tokens from each teacher adaptively. At the coarse-grained level, we summarize the knowledge from multiple teachers and transfer it to the student using a set of general-knowledge LoRA adapters with a router. Extensive experiments on various vision-language tasks demonstrate the superiority of HAWAII, compared to the popular open-source VLMs. 

---
# From Web Search towards Agentic Deep Research: Incentivizing Search with Reasoning Agents 

**Authors**: Weizhi Zhang, Yangning Li, Yuanchen Bei, Junyu Luo, Guancheng Wan, Liangwei Yang, Chenxuan Xie, Yuyao Yang, Wei-Chieh Huang, Chunyu Miao, Henry Peng Zou, Xiao Luo, Yusheng Zhao, Yankai Chen, Chunkit Chan, Peilin Zhou, Xinyang Zhang, Chenwei Zhang, Jingbo Shang, Ming Zhang, Yangqiu Song, Irwin King, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18959)  

**Abstract**: Information retrieval is a cornerstone of modern knowledge acquisition, enabling billions of queries each day across diverse domains. However, traditional keyword-based search engines are increasingly inadequate for handling complex, multi-step information needs. Our position is that Large Language Models (LLMs), endowed with reasoning and agentic capabilities, are ushering in a new paradigm termed Agentic Deep Research. These systems transcend conventional information search techniques by tightly integrating autonomous reasoning, iterative retrieval, and information synthesis into a dynamic feedback loop. We trace the evolution from static web search to interactive, agent-based systems that plan, explore, and learn. We also introduce a test-time scaling law to formalize the impact of computational depth on reasoning and search. Supported by benchmark results and the rise of open-source implementations, we demonstrate that Agentic Deep Research not only significantly outperforms existing approaches, but is also poised to become the dominant paradigm for future information seeking. All the related resources, including industry products, research papers, benchmark datasets, and open-source implementations, are collected for the community in this https URL. 

---
# A Comment On "The Illusion of Thinking": Reframing the Reasoning Cliff as an Agentic Gap 

**Authors**: Sheraz Khan, Subha Madhavan, Kannan Natarajan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18957)  

**Abstract**: The recent work by Shojaee et al. (2025), titled The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity, presents a compelling empirical finding, a reasoning cliff, where the performance of Large Reasoning Models (LRMs) collapses beyond a specific complexity threshold, which the authors posit as an intrinsic scaling limitation of Chain-of-Thought (CoT) reasoning. This commentary, while acknowledging the study's methodological rigor, contends that this conclusion is confounded by experimental artifacts. We argue that the observed failure is not evidence of a fundamental cognitive boundary, but rather a predictable outcome of system-level constraints in the static, text-only evaluation paradigm, including tool use restrictions, context window recall issues, the absence of crucial cognitive baselines, inadequate statistical reporting, and output generation limits. We reframe this performance collapse through the lens of an agentic gap, asserting that the models are not failing at reasoning, but at execution within a profoundly restrictive interface. We empirically substantiate this critique by demonstrating a striking reversal. A model, initially declaring a puzzle impossible when confined to text-only generation, now employs agentic tools to not only solve it but also master variations of complexity far beyond the reasoning cliff it previously failed to surmount. Additionally, our empirical analysis of tool-enabled models like o4-mini and GPT-4o reveals a hierarchy of agentic reasoning, from simple procedural execution to complex meta-cognitive self-correction, which has significant implications for how we define and measure machine intelligence. The illusion of thinking attributed to LRMs is less a reasoning deficit and more a consequence of an otherwise capable mind lacking the tools for action. 

---
# LLMs on a Budget? Say HOLA 

**Authors**: Zohaib Hasan Siddiqui, Jiechao Gao, Ebad Shabbir, Mohammad Anas Azeez, Rafiq Ali, Gautam Siddharth Kashyap, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2506.18952)  

**Abstract**: Running Large Language Models (LLMs) on edge devices is constrained by high compute and memory demands posing a barrier for real-time applications in sectors like healthcare, education, and embedded systems. Current solutions such as quantization, pruning, and retrieval-augmented generation (RAG) offer only partial optimizations and often compromise on speed or accuracy. We introduce HOLA, an end-to-end optimization framework for efficient LLM deployment. Internally, it leverages Hierarchical Speculative Decoding (HSD) for faster inference without quality loss. Externally, AdaComp-RAG adjusts retrieval complexity based on context needs. Together with LoBi, which blends structured pruning (LoRA) and quantization, HOLA delivers significant gains: 17.6% EMA on GSM8K, 10.5% MCA on ARC, and reduced latency and memory on edge devices like Jetson Nano--proving both scalable and production-ready. 

---
# Chain-of-Experts: Unlocking the Communication Power of Mixture-of-Experts Models 

**Authors**: Zihan Wang, Rui Pan, Jiarui Yao, Robert Csordas, Linjie Li, Lu Yin, Jiajun Wu, Tong Zhang, Manling Li, Shiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18945)  

**Abstract**: We propose Chain-of-Experts (CoE), a new Mixture-of-Experts (MoE) architecture that introduces sequential expert communication within each layer. Unlike traditional MoE models, where experts operate independently in parallel, CoE processes tokens iteratively across a chain of experts inside a layer. To support dynamic expert selection across iterations, CoE employs a dedicated router at each iteration step within a layer. This design allows tokens to re-evaluate and select different experts during each iteration, rather than being statically assigned. As a result, CoE introduces a flexible routing mechanism that increases the diversity of expert combinations and enriches the model's representational capacity. CoE demonstrates improved performance under fixed compute: on math reasoning tasks, it reduces validation loss from 1.20 to 1.12 compared to a standard MoE. Beyond performance, CoE offers a new scaling axis: depth through expert iteration, which complements conventional width/depth scaling. For example, using 2x iterations matches the performance of 3x expert selections (in width), while reducing memory usage by 17.6-42% relative to other scaling strategies. Our analysis reveals that CoE's benefits stem from its iterative residual structure and enhanced expert specialization empowered by iterative routing, which together unlock more expressive representations. Code is available at this https URL. 

---
# Mix-of-Language-Experts Architecture for Multilingual Programming 

**Authors**: Yifan Zong, Yuntian Deng, Pengyu Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.18923)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities in aiding developers with tasks like code comprehension, generation, and translation. Supporting multilingual programming -- i.e., coding tasks across multiple programming languages -- typically requires either (1) finetuning a single LLM across all programming languages, which is cost-efficient but sacrifices language-specific specialization and performance, or (2) finetuning separate LLMs for each programming language, which allows for specialization but is computationally expensive and storage-intensive due to the duplication of parameters. This paper introduces MoLE (Mix-of-Language-Experts), a novel architecture that balances efficiency and specialization for multilingual programming. MoLE is composed of a base model, a shared LoRA (low-rank adaptation) module, and a collection of language-specific LoRA modules. These modules are jointly optimized during the finetuning process, enabling effective knowledge sharing and specialization across programming languages. During inference, MoLE automatically routes to the language-specific LoRA module corresponding to the programming language of the code token being generated. Our experiments demonstrate that MoLE achieves greater parameter efficiency compared to training separate language-specific LoRAs, while outperforming a single shared LLM finetuned for all programming languages in terms of accuracy. 

---
