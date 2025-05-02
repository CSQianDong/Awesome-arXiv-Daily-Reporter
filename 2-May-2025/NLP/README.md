# Steering Large Language Models with Register Analysis for Arbitrary Style Transfer 

**Authors**: Xinchen Yang, Marine Carpuat  

**Link**: [PDF](https://arxiv.org/pdf/2505.00679)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in rewriting text across various styles. However, effectively leveraging this ability for example-based arbitrary style transfer, where an input text is rewritten to match the style of a given exemplar, remains an open challenge. A key question is how to describe the style of the exemplar to guide LLMs toward high-quality rewrites. In this work, we propose a prompting method based on register analysis to guide LLMs to perform this task. Empirical evaluations across multiple style transfer tasks show that our prompting approach enhances style transfer strength while preserving meaning more effectively than existing prompting strategies. 

---
# Rethinking Memory in AI: Taxonomy, Operations, Topics, and Future Directions 

**Authors**: Yiming Du, Wenyu Huang, Danna Zheng, Zhaowei Wang, Sebastien Montella, Mirella Lapata, Kam-Fai Wong, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00675)  

**Abstract**: Memory is a fundamental component of AI systems, underpinning large language models (LLMs) based agents. While prior surveys have focused on memory applications with LLMs, they often overlook the atomic operations that underlie memory dynamics. In this survey, we first categorize memory representations into parametric, contextual structured, and contextual unstructured and then introduce six fundamental memory operations: Consolidation, Updating, Indexing, Forgetting, Retrieval, and Compression. We systematically map these operations to the most relevant research topics across long-term, long-context, parametric modification, and multi-source memory. By reframing memory systems through the lens of atomic operations and representation types, this survey provides a structured and dynamic perspective on research, benchmark datasets, and tools related to memory in AI, clarifying the functional interplay in LLMs based agents while outlining promising directions for future research\footnote{The paper list, datasets, methods and tools are available at \href{this https URL}{this https URL\_Memory\_in\_AI}.}. 

---
# DeepCritic: Deliberate Critique with Large Language Models 

**Authors**: Wenkai Yang, Jingwen Chen, Yankai Lin, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00662)  

**Abstract**: As Large Language Models (LLMs) are rapidly evolving, providing accurate feedback and scalable oversight on their outputs becomes an urgent and critical problem. Leveraging LLMs as critique models to achieve automated supervision is a promising solution. In this work, we focus on studying and enhancing the math critique ability of LLMs. Current LLM critics provide critiques that are too shallow and superficial on each step, leading to low judgment accuracy and struggling to offer sufficient feedback for the LLM generator to correct mistakes. To tackle this issue, we propose a novel and effective two-stage framework to develop LLM critics that are capable of deliberately critiquing on each reasoning step of math solutions. In the first stage, we utilize Qwen2.5-72B-Instruct to generate 4.5K long-form critiques as seed data for supervised fine-tuning. Each seed critique consists of deliberate step-wise critiques that includes multi-perspective verifications as well as in-depth critiques of initial critiques for each reasoning step. Then, we perform reinforcement learning on the fine-tuned model with either existing human-labeled data from PRM800K or our automatically annotated data obtained via Monte Carlo sampling-based correctness estimation, to further incentivize its critique ability. Our developed critique model built on Qwen2.5-7B-Instruct not only significantly outperforms existing LLM critics (including the same-sized DeepSeek-R1-distill models and GPT-4o) on various error identification benchmarks, but also more effectively helps the LLM generator refine erroneous steps through more detailed feedback. 

---
# On the generalization of language models from in-context learning and finetuning: a controlled study 

**Authors**: Andrew K. Lampinen, Arslan Chaudhry, Stephanie C.Y. Chan, Cody Wild, Diane Wan, Alex Ku, Jörg Bornschein, Razvan Pascanu, Murray Shanahan, James L. McClelland  

**Link**: [PDF](https://arxiv.org/pdf/2505.00661)  

**Abstract**: Large language models exhibit exciting capabilities, yet can show surprisingly narrow generalization from finetuning -- from failing to generalize to simple reversals of relations they are trained on, to missing logical deductions that can be made from trained information. These failures to generalize from fine-tuning can hinder practical application of these models. However, language models' in-context learning shows different inductive biases, and can generalize better in some of these cases. Here, we explore these differences in generalization between in-context- and fine-tuning-based learning. To do so, we constructed several novel datasets to evaluate and improve models' ability to generalize from finetuning data. The datasets are constructed to isolate the knowledge in the dataset from that in pretraining, to create clean tests of generalization. We expose pretrained large models to controlled subsets of the information in these datasets -- either in context, or through fine-tuning -- and evaluate their performance on test sets that require various types of generalization. We find overall that in data-matched settings, in-context learning can generalize more flexibly than fine-tuning (though we also find some qualifications of prior findings, such as cases when fine-tuning can generalize to reversals embedded in a larger structure of knowledge). We build on these findings to propose a method to enable improved generalization from fine-tuning: adding in-context inferences to finetuning data. We show that this method improves generalization across various splits of our datasets and other benchmarks. Our results have implications for understanding the inductive biases of different modes of learning in language models, and practically improving their performance. 

---
# Large Language Models Understanding: an Inherent Ambiguity Barrier 

**Authors**: Daniel N. Nissani  

**Link**: [PDF](https://arxiv.org/pdf/2505.00654)  

**Abstract**: A lively ongoing debate is taking place, since the extraordinary emergence of Large Language Models (LLMs) with regards to their capability to understand the world and capture the meaning of the dialogues in which they are involved. Arguments and counter-arguments have been proposed based upon thought experiments, anecdotal conversations between LLMs and humans, statistical linguistic analysis, philosophical considerations, and more. In this brief paper we present a counter-argument based upon a thought experiment and semi-formal considerations leading to an inherent ambiguity barrier which prevents LLMs from having any understanding of what their amazingly fluent dialogues mean. 

---
# The Illusion of Role Separation: Hidden Shortcuts in LLM Role Learning (and How to Fix Them) 

**Authors**: Zihao Wang, Yibo Jiang, Jiahao Yu, Heqing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00626)  

**Abstract**: Large language models (LLMs) that integrate multiple input roles (e.g., system instructions, user queries, external tool outputs) are increasingly prevalent in practice. Ensuring that the model accurately distinguishes messages from each role -- a concept we call \emph{role separation} -- is crucial for consistent multi-role behavior. Although recent work often targets state-of-the-art prompt injection defenses, it remains unclear whether such methods truly teach LLMs to differentiate roles or merely memorize known triggers. In this paper, we examine \emph{role-separation learning}: the process of teaching LLMs to robustly distinguish system and user tokens. Through a \emph{simple, controlled experimental framework}, we find that fine-tuned models often rely on two proxies for role identification: (1) task type exploitation, and (2) proximity to begin-of-text. Although data augmentation can partially mitigate these shortcuts, it generally leads to iterative patching rather than a deeper fix. To address this, we propose reinforcing \emph{invariant signals} that mark role boundaries by adjusting token-wise cues in the model's input encoding. In particular, manipulating position IDs helps the model learn clearer distinctions and reduces reliance on superficial proxies. By focusing on this mechanism-centered perspective, our work illuminates how LLMs can more reliably maintain consistent multi-role behavior without merely memorizing known prompts or triggers. 

---
# FineScope : Precision Pruning for Domain-Specialized Large Language Models Using SAE-Guided Self-Data Cultivation 

**Authors**: Chaitali Bhattacharyya, Yeseong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.00624)  

**Abstract**: Training large language models (LLMs) from scratch requires significant computational resources, driving interest in developing smaller, domain-specific LLMs that maintain both efficiency and strong task performance. Medium-sized models such as LLaMA, llama} have served as starting points for domain-specific adaptation, but they often suffer from accuracy degradation when tested on specialized datasets. We introduce FineScope, a framework for deriving compact, domain-optimized LLMs from larger pretrained models. FineScope leverages the Sparse Autoencoder (SAE) framework, inspired by its ability to produce interpretable feature representations, to extract domain-specific subsets from large datasets. We apply structured pruning with domain-specific constraints, ensuring that the resulting pruned models retain essential knowledge for the target domain. To further enhance performance, these pruned models undergo self-data distillation, leveraging SAE-curated datasets to restore key domain-specific information lost during pruning. Extensive experiments and ablation studies demonstrate that FineScope achieves highly competitive performance, outperforming several large-scale state-of-the-art LLMs in domain-specific tasks. Additionally, our results show that FineScope enables pruned models to regain a substantial portion of their original performance when fine-tuned with SAE-curated datasets. Furthermore, applying these datasets to fine-tune pretrained LLMs without pruning also improves their domain-specific accuracy, highlighting the robustness of our approach. The code will be released. 

---
# Block Circulant Adapter for Large Language Models 

**Authors**: Xinyu Ding, Meiqi Wang, Siyu Liao, Zhongfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00582)  

**Abstract**: Fine-tuning large language models (LLMs) is difficult due to their huge model size. Recent Fourier domain-based methods show potential for reducing fine-tuning costs. We propose a block circulant matrix-based fine-tuning method with a stable training heuristic to leverage the properties of circulant matrices and one-dimensional Fourier transforms to reduce storage and computation costs. Experiments show that our method uses $14\times$ less number of parameters than VeRA, $16\times$ smaller than LoRA and $32\times$ less FLOPs than FourierFT, while maintaining close or better task performance. Our approach presents a promising way in frequency domain to fine-tune large models on downstream tasks. 

---
# FreqKV: Frequency Domain Key-Value Compression for Efficient Context Window Extension 

**Authors**: Jushi Kai, Boyi Zeng, Yixuan Wang, Haoli Bai, Bo Jiang, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.00570)  

**Abstract**: Extending the context window in large language models (LLMs) is essential for applications involving long-form content generation. However, the linear increase in key-value (KV) cache memory requirements and the quadratic complexity of self-attention with respect to sequence length present significant challenges during fine-tuning and inference. Existing methods suffer from performance degradation when extending to longer contexts. In this work, we introduce a novel context extension method that optimizes both fine-tuning and inference efficiency. Our method exploits a key observation: in the frequency domain, the energy distribution of the KV cache is primarily concentrated in low-frequency components. By filtering out the high-frequency components, the KV cache can be effectively compressed with minimal information loss. Building on this insight, we propose an efficient compression technique, FreqKV, that iteratively compresses the increasing KV cache to a fixed size in the frequency domain, applicable to both fine-tuning and inference. FreqKV introduces no additional parameters or architectural modifications. With minimal fine-tuning, LLMs can learn to leverage the limited cache that is compressed in the frequency domain and extend the context window efficiently. Experiments on various long context language modeling and understanding tasks demonstrate the efficiency and efficacy of the proposed method. 

---
# Triggering Hallucinations in LLMs: A Quantitative Study of Prompt-Induced Hallucination in Large Language Models 

**Authors**: Makoto Sato  

**Link**: [PDF](https://arxiv.org/pdf/2505.00557)  

**Abstract**: Hallucinations in large language models (LLMs) present a growing challenge across real-world applications, from healthcare to law, where factual reliability is essential. Despite advances in alignment and instruction tuning, LLMs can still generate outputs that are fluent yet fundamentally untrue. Understanding the cognitive dynamics that underlie these hallucinations remains an open problem. In this study, we propose a prompt-based framework to systematically trigger and quantify hallucination: a Hallucination-Inducing Prompt (HIP), which synthetically fuses semantically distant concepts (e.g., periodic table of elements and tarot divination) in a misleading way, and a Hallucination Quantifying Prompt (HQP), which scores the plausibility, confidence, and coherence of the output. Controlled experiments across multiple LLMs revealed that HIPs consistently produced less coherent and more hallucinated responses than their null-fusion controls. These effects varied across models, with reasoning-oriented LLMs showing distinct profiles from general-purpose ones. Our framework provides a reproducible testbed for studying hallucination vulnerability, and opens the door to developing safer, more introspective LLMs that can detect and self-regulate the onset of conceptual instability. 

---
# 100 Days After DeepSeek-R1: A Survey on Replication Studies and More Directions for Reasoning Language Models 

**Authors**: Chong Zhang, Yue Deng, Xiang Lin, Bin Wang, Dianwen Ng, Hai Ye, Xingxuan Li, Yao Xiao, Zhanfeng Mo, Qi Zhang, Lidong Bing  

**Link**: [PDF](https://arxiv.org/pdf/2505.00551)  

**Abstract**: The recent development of reasoning language models (RLMs) represents a novel evolution in large language models. In particular, the recent release of DeepSeek-R1 has generated widespread social impact and sparked enthusiasm in the research community for exploring the explicit reasoning paradigm of language models. However, the implementation details of the released models have not been fully open-sourced by DeepSeek, including DeepSeek-R1-Zero, DeepSeek-R1, and the distilled small models. As a result, many replication studies have emerged aiming to reproduce the strong performance achieved by DeepSeek-R1, reaching comparable performance through similar training procedures and fully open-source data resources. These works have investigated feasible strategies for supervised fine-tuning (SFT) and reinforcement learning from verifiable rewards (RLVR), focusing on data preparation and method design, yielding various valuable insights. In this report, we provide a summary of recent replication studies to inspire future research. We primarily focus on SFT and RLVR as two main directions, introducing the details for data construction, method design and training procedure of current replication studies. Moreover, we conclude key findings from the implementation details and experimental results reported by these studies, anticipating to inspire future research. We also discuss additional techniques of enhancing RLMs, highlighting the potential of expanding the application scope of these models, and discussing the challenges in development. By this survey, we aim to help researchers and developers of RLMs stay updated with the latest advancements, and seek to inspire new ideas to further enhance RLMs. 

---
# HalluMix: A Task-Agnostic, Multi-Domain Benchmark for Real-World Hallucination Detection 

**Authors**: Deanna Emery, Michael Goitia, Freddie Vargus, Iulia Neagu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00506)  

**Abstract**: As large language models (LLMs) are increasingly deployed in high-stakes domains, detecting hallucinated content$\unicode{x2013}$text that is not grounded in supporting evidence$\unicode{x2013}$has become a critical challenge. Existing benchmarks for hallucination detection are often synthetically generated, narrowly focused on extractive question answering, and fail to capture the complexity of real-world scenarios involving multi-document contexts and full-sentence outputs. We introduce the HalluMix Benchmark, a diverse, task-agnostic dataset that includes examples from a range of domains and formats. Using this benchmark, we evaluate seven hallucination detection systems$\unicode{x2013}$both open and closed source$\unicode{x2013}$highlighting differences in performance across tasks, document lengths, and input representations. Our analysis highlights substantial performance disparities between short and long contexts, with critical implications for real-world Retrieval Augmented Generation (RAG) implementations. Quotient Detections achieves the best overall performance, with an accuracy of 0.82 and an F1 score of 0.84. 

---
# Computational Identification of Regulatory Statements in EU Legislation 

**Authors**: Gijs Jan Brandsma, Jens Blom-Hansen, Christiaan Meijer, Kody Moodley  

**Link**: [PDF](https://arxiv.org/pdf/2505.00479)  

**Abstract**: Identifying regulatory statements in legislation is useful for developing metrics to measure the regulatory density and strictness of legislation. A computational method is valuable for scaling the identification of such statements from a growing body of EU legislation, constituting approximately 180,000 published legal acts between 1952 and 2023. Past work on extraction of these statements varies in the permissiveness of their definitions for what constitutes a regulatory statement. In this work, we provide a specific definition for our purposes based on the institutional grammar tool. We develop and compare two contrasting approaches for automatically identifying such statements in EU legislation, one based on dependency parsing, and the other on a transformer-based machine learning model. We found both approaches performed similarly well with accuracies of 80% and 84% respectively and a K alpha of 0.58. The high accuracies and not exceedingly high agreement suggests potential for combining strengths of both approaches. 

---
# Red Teaming Large Language Models for Healthcare 

**Authors**: Vahid Balazadeh, Michael Cooper, David Pellow, Atousa Assadi, Jennifer Bell, Jim Fackler, Gabriel Funingana, Spencer Gable-Cook, Anirudh Gangadhar, Abhishek Jaiswal, Sumanth Kaja, Christopher Khoury, Randy Lin, Kaden McKeen, Sara Naimimohasses, Khashayar Namdar, Aviraj Newatia, Allan Pang, Anshul Pattoo, Sameer Peesapati, Diana Prepelita, Bogdana Rakova, Saba Sadatamin, Rafael Schulman, Ajay Shah, Syed Azhar Shah, Syed Ahmar Shah, Babak Taati, Balagopal Unnikrishnan, Stephanie Williams, Rahul G Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00467)  

**Abstract**: We present the design process and findings of the pre-conference workshop at the Machine Learning for Healthcare Conference (2024) entitled Red Teaming Large Language Models for Healthcare, which took place on August 15, 2024. Conference participants, comprising a mix of computational and clinical expertise, attempted to discover vulnerabilities -- realistic clinical prompts for which a large language model (LLM) outputs a response that could cause clinical harm. Red-teaming with clinicians enables the identification of LLM vulnerabilities that may not be recognised by LLM developers lacking clinical expertise. We report the vulnerabilities found, categorise them, and present the results of a replication study assessing the vulnerabilities across all LLMs provided. 

---
# CSE-SFP: Enabling Unsupervised Sentence Representation Learning via a Single Forward Pass 

**Authors**: Bowen Zhang, Zixin Song, Chunping Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.00389)  

**Abstract**: As a fundamental task in Information Retrieval and Computational Linguistics, sentence representation has profound implications for a wide range of practical applications such as text clustering, content analysis, question-answering systems, and web search. Recent advances in pre-trained language models (PLMs) have driven remarkable progress in this field, particularly through unsupervised embedding derivation methods centered on discriminative PLMs like BERT. However, due to time and computational constraints, few efforts have attempted to integrate unsupervised sentence representation with generative PLMs, which typically possess much larger parameter sizes. Given that state-of-the-art models in both academia and industry are predominantly based on generative architectures, there is a pressing need for an efficient unsupervised text representation framework tailored to decoder-only PLMs. To address this concern, we propose CSE-SFP, an innovative method that exploits the structural characteristics of generative models. Compared to existing strategies, CSE-SFP requires only a single forward pass to perform effective unsupervised contrastive learning. Rigorous experimentation demonstrates that CSE-SFP not only produces higher-quality embeddings but also significantly reduces both training time and memory consumption. Furthermore, we introduce two ratio metrics that jointly assess alignment and uniformity, thereby providing a more robust means for evaluating the semantic spatial properties of encoding models. 

---
# KoACD: The First Korean Adolescent Dataset for Cognitive Distortion Analysis 

**Authors**: JunSeo Kim, HyeHyeon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.00367)  

**Abstract**: Cognitive distortion refers to negative thinking patterns that can lead to mental health issues like depression and anxiety in adolescents. Previous studies using natural language processing (NLP) have focused mainly on small-scale adult datasets, with limited research on adolescents. This study introduces KoACD, the first large-scale dataset of cognitive distortions in Korean adolescents, containing 108,717 instances. We applied a multi-Large Language Model (LLM) negotiation method to refine distortion classification and generate synthetic data using two approaches: cognitive clarification for textual clarity and cognitive balancing for diverse distortion representation. Validation through LLMs and expert evaluations showed that while LLMs classified distortions with explicit markers, they struggled with context-dependent reasoning, where human evaluators demonstrated higher accuracy. KoACD aims to enhance future research on cognitive distortion detection. 

---
# Enhancing AI-Driven Education: Integrating Cognitive Frameworks, Linguistic Feedback Analysis, and Ethical Considerations for Improved Content Generation 

**Authors**: Antoun Yaacoub, Sansiri Tarnpradab, Phattara Khumprom, Zainab Assaghir, Lionel Prevost, Jérôme Da-Rugna  

**Link**: [PDF](https://arxiv.org/pdf/2505.00339)  

**Abstract**: Artificial intelligence (AI) is rapidly transforming education, presenting unprecedented opportunities for personalized learning and streamlined content creation. However, realizing the full potential of AI in educational settings necessitates careful consideration of the quality, cognitive depth, and ethical implications of AI-generated materials. This paper synthesizes insights from four related studies to propose a comprehensive framework for enhancing AI-driven educational tools. We integrate cognitive assessment frameworks (Bloom's Taxonomy and SOLO Taxonomy), linguistic analysis of AI-generated feedback, and ethical design principles to guide the development of effective and responsible AI tools. We outline a structured three-phase approach encompassing cognitive alignment, linguistic feedback integration, and ethical safeguards. The practical application of this framework is demonstrated through its integration into OneClickQuiz, an AI-powered Moodle plugin for quiz generation. This work contributes a comprehensive and actionable guide for educators, researchers, and developers aiming to harness AI's potential while upholding pedagogical and ethical standards in educational content generation. 

---
# Consistency in Language Models: Current Landscape, Challenges, and Future Directions 

**Authors**: Jekaterina Novikova, Carol Anderson, Borhane Blili-Hamelin, Subhabrata Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2505.00268)  

**Abstract**: The hallmark of effective language use lies in consistency -- expressing similar meanings in similar contexts and avoiding contradictions. While human communication naturally demonstrates this principle, state-of-the-art language models struggle to maintain reliable consistency across different scenarios. This paper examines the landscape of consistency research in AI language systems, exploring both formal consistency (including logical rule adherence) and informal consistency (such as moral and factual coherence). We analyze current approaches to measure aspects of consistency, identify critical research gaps in standardization of definitions, multilingual assessment, and methods to improve consistency. Our findings point to an urgent need for robust benchmarks to measure and interdisciplinary approaches to ensure consistency in the application of language models on domain-specific tasks while preserving the utility and adaptability. 

---
# Enriching the Korean Learner Corpus with Multi-reference Annotations and Rubric-Based Scoring 

**Authors**: Jayoung Song, KyungTae Lim, Jungyeul Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.00261)  

**Abstract**: Despite growing global interest in Korean language education, there remains a significant lack of learner corpora tailored to Korean L2 writing. To address this gap, we enhance the KoLLA Korean learner corpus by adding multiple grammatical error correction (GEC) references, thereby enabling more nuanced and flexible evaluation of GEC systems, and reflects the variability of human language. Additionally, we enrich the corpus with rubric-based scores aligned with guidelines from the Korean National Language Institute, capturing grammatical accuracy, coherence, and lexical diversity. These enhancements make KoLLA a robust and standardized resource for research in Korean L2 education, supporting advancements in language learning, assessment, and automated error correction. 

---
# IP-CRR: Information Pursuit for Interpretable Classification of Chest Radiology Reports 

**Authors**: Yuyan Ge, Kwan Ho Ryan Chan, Pablo Messina, René Vidal  

**Link**: [PDF](https://arxiv.org/pdf/2505.00191)  

**Abstract**: The development of AI-based methods for analyzing radiology reports could lead to significant advances in medical diagnosis--from improving diagnostic accuracy to enhancing efficiency and reducing workload. However, the lack of interpretability in these methods has hindered their adoption in clinical settings. In this paper, we propose an interpretable-by-design framework for classifying radiology reports. The key idea is to extract a set of most informative queries from a large set of reports and use these queries and their corresponding answers to predict a diagnosis. Thus, the explanation for a prediction is, by construction, the set of selected queries and answers. We use the Information Pursuit framework to select informative queries, the Flan-T5 model to determine if facts are present in the report, and a classifier to predict the disease. Experiments on the MIMIC-CXR dataset demonstrate the effectiveness of the proposed method, highlighting its potential to enhance trust and usability in medical AI. 

---
# AdaptMI: Adaptive Skill-based In-context Math Instruction for Small Language Models 

**Authors**: Yinghui He, Abhishek Panigrahi, Yong Lin, Sanjeev Arora  

**Link**: [PDF](https://arxiv.org/pdf/2505.00147)  

**Abstract**: In-context learning (ICL) allows a language model to improve its problem-solving capability when provided with suitable information in context. Since the choice of in-context information can be determined based on the problem itself, in-context learning is analogous to human learning from teachers in a classroom. Recent works (Didolkar et al., 2024a; 2024b) show that ICL performance can be improved by leveraging a frontier large language model's (LLM) ability to predict required skills to solve a problem, popularly referred to as an LLM's metacognition, and using the recommended skills to construct necessary in-context examples. While this skill-based strategy boosts ICL performance in larger models, its gains on small language models (SLMs) have been minimal, highlighting a performance gap in ICL capabilities. We investigate this gap and show that skill-based prompting can hurt SLM performance on easy questions by introducing unnecessary information, akin to cognitive overload. To address this, we introduce AdaptMI, an adaptive approach to selecting skill-based in-context Math Instructions for SLMs. Inspired by cognitive load theory from human pedagogy, our method only introduces skill-based examples when the model performs poorly. We further propose AdaptMI+, which adds examples targeted to the specific skills missing from the model's responses. On 5-shot evaluations across popular math benchmarks and five SLMs (1B--7B; Qwen, Llama), AdaptMI+ improves accuracy by up to 6% over naive skill-based strategies. 

---
# Between Underthinking and Overthinking: An Empirical Study of Reasoning Length and correctness in LLMs 

**Authors**: Jinyan Su, Jennifer Healey, Preslav Nakov, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2505.00127)  

**Abstract**: Large language models (LLMs) are increasingly optimized for long reasoning, under the assumption that more reasoning leads to better performance. However, emerging evidence suggests that longer responses can sometimes degrade accuracy rather than improve it. In this paper, we conduct a systematic empirical study of the relationship between reasoning length and answer correctness. We find that LLMs tend to overthink simple problems, generating unnecessarily long outputs, and underthink harder ones, failing to extend their reasoning when it is most needed. This indicates that models might misjudge problem difficulty and fail to calibrate their response length appropriately. Furthermore, we investigate the effects of length reduction with a preference optimization algorithm when simply preferring the shorter responses regardless of answer correctness. Experiments show that the generation length can be significantly reduced while maintaining acceptable accuracy. Our findings highlight generation length as a meaningful signal for reasoning behavior and motivate further exploration into LLMs' self-awareness in reasoning length adaptation. 

---
# Fine-Tuning LLMs for Low-Resource Dialect Translation: The Case of Lebanese 

**Authors**: Silvana Yakhni, Ali Chehab  

**Link**: [PDF](https://arxiv.org/pdf/2505.00114)  

**Abstract**: This paper examines the effectiveness of Large Language Models (LLMs) in translating the low-resource Lebanese dialect, focusing on the impact of culturally authentic data versus larger translated datasets. We compare three fine-tuning approaches: Basic, contrastive, and grammar-hint tuning, using open-source Aya23 models. Experiments reveal that models fine-tuned on a smaller but culturally aware Lebanese dataset (LW) consistently outperform those trained on larger, non-native data. The best results were achieved through contrastive fine-tuning paired with contrastive prompting, which indicates the benefits of exposing translation models to bad examples. In addition, to ensure authentic evaluation, we introduce LebEval, a new benchmark derived from native Lebanese content, and compare it to the existing FLoRes benchmark. Our findings challenge the "More Data is Better" paradigm and emphasize the crucial role of cultural authenticity in dialectal translation. We made our datasets and code available on Github. 

---
# ConSens: Assessing context grounding in open-book question answering 

**Authors**: Ivan Vankov, Matyo Ivanov, Adriana Correia, Victor Botev  

**Link**: [PDF](https://arxiv.org/pdf/2505.00065)  

**Abstract**: Large Language Models (LLMs) have demonstrated considerable success in open-book question answering (QA), where the task requires generating answers grounded in a provided external context. A critical challenge in open-book QA is to ensure that model responses are based on the provided context rather than its parametric knowledge, which can be outdated, incomplete, or incorrect. Existing evaluation methods, primarily based on the LLM-as-a-judge approach, face significant limitations, including biases, scalability issues, and dependence on costly external systems. To address these challenges, we propose a novel metric that contrasts the perplexity of the model response under two conditions: when the context is provided and when it is not. The resulting score quantifies the extent to which the model's answer relies on the provided context. The validity of this metric is demonstrated through a series of experiments that show its effectiveness in identifying whether a given answer is grounded in the provided context. Unlike existing approaches, this metric is computationally efficient, interpretable, and adaptable to various use cases, offering a scalable and practical solution to assess context utilization in open-book QA systems. 

---
# GDI-Bench: A Benchmark for General Document Intelligence with Vision and Reasoning Decoupling 

**Authors**: Siqi Li, Yufan Shen, Xiangnan Chen, Jiayi Chen, Hengwei Ju, Haodong Duan, Song Mao, Hongbin Zhou, Bo Zhang, Pinlong Cai, Licheng Wen, Botian Shi, Yong Liu, Xinyu Cai, Yu Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.00063)  

**Abstract**: The rapid advancement of multimodal large language models (MLLMs) has profoundly impacted the document domain, creating a wide array of application scenarios. This progress highlights the need for a comprehensive benchmark to evaluate these models' capabilities across various document-specific tasks. However, existing benchmarks often fail to locate specific model weaknesses or guide systematic improvements. To bridge this gap, we introduce a General Document Intelligence Benchmark (GDI-Bench), featuring 1.9k images across 9 key scenarios and 19 document-specific tasks. By decoupling visual complexity and reasoning complexity, the GDI-Bench structures graded tasks that allow performance assessment by difficulty, aiding in model weakness identification and optimization guidance. We evaluate the GDI-Bench on various open-source and closed-source models, conducting decoupled analyses in the visual and reasoning domains. For instance, the GPT-4o model excels in reasoning tasks but exhibits limitations in visual capabilities. To address the diverse tasks and domains in the GDI-Bench, we propose a GDI Model that mitigates the issue of catastrophic forgetting during the supervised fine-tuning (SFT) process through a intelligence-preserving training strategy. Our model achieves state-of-the-art performance on previous benchmarks and the GDI-Bench. Both our benchmark and model will be open source. 

---
# Enhancing Security and Strengthening Defenses in Automated Short-Answer Grading Systems 

**Authors**: Sahar Yarmohammadtoosky, Yiyun Zhou, Victoria Yaneva, Peter Baldwin, Saed Rezayi, Brian Clauser, Polina Harikeo  

**Link**: [PDF](https://arxiv.org/pdf/2505.00061)  

**Abstract**: This study examines vulnerabilities in transformer-based automated short-answer grading systems used in medical education, with a focus on how these systems can be manipulated through adversarial gaming strategies. Our research identifies three main types of gaming strategies that exploit the system's weaknesses, potentially leading to false positives. To counteract these vulnerabilities, we implement several adversarial training methods designed to enhance the systems' robustness. Our results indicate that these methods significantly reduce the susceptibility of grading systems to such manipulations, especially when combined with ensemble techniques like majority voting and ridge regression, which further improve the system's defense against sophisticated adversarial inputs. Additionally, employing large language models such as GPT-4 with varied prompting techniques has shown promise in recognizing and scoring gaming strategies effectively. The findings underscore the importance of continuous improvements in AI-driven educational tools to ensure their reliability and fairness in high-stakes settings. 

---
# Fact-Consistency Evaluation of Text-to-SQL Generation for Business Intelligence Using Exaone 3.5 

**Authors**: Jeho Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.00060)  

**Abstract**: Large Language Models (LLMs) have shown promise in enabling natural language interfaces for structured data querying through text-to-SQL generation. However, their application in real-world Business Intelligence (BI) contexts remains limited due to semantic hallucinations, structural errors, and a lack of domain-specific evaluation frameworks. In this study, we propose a Fact-Consistency Evaluation Framework for assessing the semantic accuracy of LLM-generated SQL outputs using Exaone 3.5--an instruction-tuned, bilingual LLM optimized for enterprise tasks. We construct a domain-specific benchmark comprising 219 natural language business questions across five SQL complexity levels, derived from actual sales data in LG Electronics' internal BigQuery environment. Each question is paired with a gold-standard SQL query and a validated ground-truth answer. We evaluate model performance using answer accuracy, execution success rate, semantic error rate, and non-response rate. Experimental results show that while Exaone 3.5 performs well on simple aggregation tasks (93% accuracy in L1), it exhibits substantial degradation in arithmetic reasoning (4% accuracy in H1) and grouped ranking tasks (31% in H4), with semantic errors and non-responses concentrated in complex cases. Qualitative error analysis further identifies common failure types such as misapplied arithmetic logic, incomplete filtering, and incorrect grouping operations. Our findings highlight the current limitations of LLMs in business-critical environments and underscore the need for fact-consistency validation layers and hybrid reasoning approaches. This work contributes a reproducible benchmark and evaluation methodology for advancing reliable natural language interfaces to structured enterprise data systems. 

---
# BERSting at the Screams: A Benchmark for Distanced, Emotional and Shouted Speech Recognition 

**Authors**: Paige Tuttösí, Mantaj Dhillon, Luna Sang, Shane Eastwood, Poorvi Bhatia, Quang Minh Dinh, Avni Kapoor, Yewon Jin, Angelica Lim  

**Link**: [PDF](https://arxiv.org/pdf/2505.00059)  

**Abstract**: Some speech recognition tasks, such as automatic speech recognition (ASR), are approaching or have reached human performance in many reported metrics. Yet, they continue to struggle in complex, real-world, situations, such as with distanced speech. Previous challenges have released datasets to address the issue of distanced ASR, however, the focus remains primarily on distance, specifically relying on multi-microphone array systems. Here we present the B(asic) E(motion) R(andom phrase) S(hou)t(s) (BERSt) dataset. The dataset contains almost 4 hours of English speech from 98 actors with varying regional and non-native accents. The data was collected on smartphones in the actors homes and therefore includes at least 98 different acoustic environments. The data also includes 7 different emotion prompts and both shouted and spoken utterances. The smartphones were places in 19 different positions, including obstructions and being in a different room than the actor. This data is publicly available for use and can be used to evaluate a variety of speech recognition tasks, including: ASR, shout detection, and speech emotion recognition (SER). We provide initial benchmarks for ASR and SER tasks, and find that ASR degrades both with an increase in distance and shout level and shows varied performance depending on the intended emotion. Our results show that the BERSt dataset is challenging for both ASR and SER tasks and continued work is needed to improve the robustness of such systems for more accurate real-world use. 

---
# A Report on the llms evaluating the high school questions 

**Authors**: Zhu Jiawei, Chen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.00057)  

**Abstract**: This report aims to evaluate the performance of large language models (LLMs) in solving high school science questions and to explore their potential applications in the educational field. With the rapid development of LLMs in the field of natural language processing, their application in education has attracted widespread attention. This study selected mathematics exam questions from the college entrance examinations (2019-2023) as evaluation data and utilized at least eight LLM APIs to provide answers. A comprehensive assessment was conducted based on metrics such as accuracy, response time, logical reasoning, and creativity. Through an in-depth analysis of the evaluation results, this report reveals the strengths and weaknesses of LLMs in handling high school science questions and discusses their implications for educational practice. The findings indicate that although LLMs perform excellently in certain aspects, there is still room for improvement in logical reasoning and creative problem-solving. This report provides an empirical foundation for further research and application of LLMs in the educational field and offers suggestions for improvement. 

---
# Clustering Internet Memes Through Template Matching and Multi-Dimensional Similarity 

**Authors**: Tygo Bloem, Filip Ilievski  

**Link**: [PDF](https://arxiv.org/pdf/2505.00056)  

**Abstract**: Meme clustering is critical for toxicity detection, virality modeling, and typing, but it has received little attention in previous research. Clustering similar Internet memes is challenging due to their multimodality, cultural context, and adaptability. Existing approaches rely on databases, overlook semantics, and struggle to handle diverse dimensions of similarity. This paper introduces a novel method that uses template-based matching with multi-dimensional similarity features, thus eliminating the need for predefined databases and supporting adaptive matching. Memes are clustered using local and global features across similarity categories such as form, visual content, text, and identity. Our combined approach outperforms existing clustering methods, producing more consistent and coherent clusters, while similarity-based feature sets enable adaptability and align with human intuition. We make all supporting code publicly available to support subsequent research. Code: this https URL 

---
# Emotional Analysis of Fashion Trends Using Social Media and AI: Sentiment Analysis on Twitter for Fashion Trend Forecasting 

**Authors**: Aayam Bansal, Agneya Tharun  

**Link**: [PDF](https://arxiv.org/pdf/2505.00050)  

**Abstract**: This study explores the intersection of fashion trends and social media sentiment through computational analysis of Twitter data using the T4SA (Twitter for Sentiment Analysis) dataset. By applying natural language processing and machine learning techniques, we examine how sentiment patterns in fashion-related social media conversations can serve as predictors for emerging fashion trends. Our analysis involves the identification and categorization of fashion-related content, sentiment classification with improved normalization techniques, time series decomposition, statistically validated causal relationship modeling, cross-platform sentiment comparison, and brand-specific sentiment analysis. Results indicate correlations between sentiment patterns and fashion theme popularity, with accessories and streetwear themes showing statistically significant rising trends. The Granger causality analysis establishes sustainability and streetwear as primary trend drivers, showing bidirectional relationships with several other themes. The findings demonstrate that social media sentiment analysis can serve as an effective early indicator of fashion trend trajectories when proper statistical validation is applied. Our improved predictive model achieved 78.35% balanced accuracy in sentiment classification, establishing a reliable foundation for trend prediction across positive, neutral, and negative sentiment categories. 

---
# Base Models Beat Aligned Models at Randomness and Creativity 

**Authors**: Peter West, Christopher Potts  

**Link**: [PDF](https://arxiv.org/pdf/2505.00047)  

**Abstract**: Alignment has quickly become a default ingredient in LLM development, with techniques such as reinforcement learning from human feedback making models act safely, follow instructions, and perform ever-better on complex tasks. While these techniques are certainly useful, we propose that they should not be universally applied and demonstrate a range of tasks on which base language models consistently outperform their popular aligned forms. Particularly, we study tasks that require unpredictable outputs, such as random number generation, mixed strategy games (rock-paper-scissors and hide-and-seek), and creative writing. In each case, aligned models tend towards narrow behaviors that result in distinct disadvantages, for instance, preferring to generate "7" over other uniformly random numbers, becoming almost fully predictable in some game states, or prioritizing pleasant writing over creative originality. Across models tested, better performance on common benchmarks tends to correlate with worse performance on our tasks, suggesting an effective trade-off in the required capabilities. 

---
# Graph RAG for Legal Norms: A Hierarchical and Temporal Approach 

**Authors**: Hudson de Martim  

**Link**: [PDF](https://arxiv.org/pdf/2505.00039)  

**Abstract**: This article proposes an adaptation of Graph Retrieval Augmented Generation (Graph RAG) specifically designed for the analysis and comprehension of legal norms, which are characterized by their predefined hierarchical structure, extensive network of internal and external references and multiple temporal versions. By combining structured knowledge graphs with contextually enriched text segments, Graph RAG offers a promising solution to address the inherent complexity and vast volume of legal data. The integration of hierarchical structure and temporal evolution into knowledge graphs - along with the concept of comprehensive Text Units - facilitates the construction of richer, interconnected representations of legal knowledge. Through a detailed analysis of Graph RAG and its application to legal norm datasets, this article aims to significantly advance the field of Artificial Intelligence applied to Law, creating opportunities for more effective systems in legal research, legislative analysis, and decision support. 

---
# HyPerAlign: Hypotheses-driven Personalized Alignment 

**Authors**: Cristina Garbacea, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00038)  

**Abstract**: Alignment algorithms are widely used to align large language models (LLMs) to human users based on preference annotations that reflect their intended real-world use cases. Typically these (often divergent) preferences are aggregated over a diverse set of users, resulting in fine-tuned models that are aligned to the ``average-user'' preference. Nevertheless, current models are used by individual users in very specific contexts and situations, emphasizing the need for user-dependent preference control. In this work we address the problem of personalizing LLM outputs to their users, aiming to generate customized responses tailored to individual users, instead of generic outputs that emulate the collective voices of diverse populations. We propose a novel interpretable and sample-efficient hypotheses-driven personalization approach (HyPerAlign) where given few-shot examples written by a particular user, we first infer hypotheses about their communication strategies, personality and writing style, then prompt LLM models with these hypotheses and user specific attributes to generate customized outputs. We conduct experiments on two different personalization tasks, authorship attribution and deliberative alignment, with datasets from diverse domains (news articles, blog posts, emails, jailbreaking benchmarks), and demonstrate the superiority of hypotheses-driven personalization approach when compared to preference-based fine-tuning methods. For deliberative alignment, the helpfulness of LLM models is improved by up to $70\%$ on average. For authorship attribution, results indicate consistently high win-rates (commonly $>90\%$) against state-of-the-art preference fine-tuning approaches for LLM personalization across diverse user profiles and LLM models. Overall, our approach represents an interpretable and sample-efficient strategy for the personalization of LLM models to individual users. 

---
# A Framework to Assess the Persuasion Risks Large Language Model Chatbots Pose to Democratic Societies 

**Authors**: Zhongren Chen, Joshua Kalla, Quan Le, Shinpei Nakamura-Sakai, Jasjeet Sekhon, Ruixiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00036)  

**Abstract**: In recent years, significant concern has emerged regarding the potential threat that Large Language Models (LLMs) pose to democratic societies through their persuasive capabilities. We expand upon existing research by conducting two survey experiments and a real-world simulation exercise to determine whether it is more cost effective to persuade a large number of voters using LLM chatbots compared to standard political campaign practice, taking into account both the "receive" and "accept" steps in the persuasion process (Zaller 1992). These experiments improve upon previous work by assessing extended interactions between humans and LLMs (instead of using single-shot interactions) and by assessing both short- and long-run persuasive effects (rather than simply asking users to rate the persuasiveness of LLM-produced content). In two survey experiments (N = 10,417) across three distinct political domains, we find that while LLMs are about as persuasive as actual campaign ads once voters are exposed to them, political persuasion in the real-world depends on both exposure to a persuasive message and its impact conditional on exposure. Through simulations based on real-world parameters, we estimate that LLM-based persuasion costs between \$48-\$74 per persuaded voter compared to \$100 for traditional campaign methods, when accounting for the costs of exposure. However, it is currently much easier to scale traditional campaign persuasion methods than LLM-based persuasion. While LLMs do not currently appear to have substantially greater potential for large-scale political persuasion than existing non-LLM methods, this may change as LLM capabilities continue to improve and it becomes easier to scalably encourage exposure to persuasive LLMs. 

---
# Linguistic Complexity and Socio-cultural Patterns in Hip-Hop Lyrics 

**Authors**: Aayam Bansal, Raghav Agarwal, Kaashvi Jain  

**Link**: [PDF](https://arxiv.org/pdf/2505.00035)  

**Abstract**: This paper presents a comprehensive computational framework for analyzing linguistic complexity and socio-cultural trends in hip-hop lyrics. Using a dataset of 3,814 songs from 146 influential artists spanning four decades (1980-2020), we employ natural language processing techniques to quantify multiple dimensions of lyrical complexity. Our analysis reveals a 23.7% increase in vocabulary diversity over the study period, with East Coast artists demonstrating 17.3% higher lexical variation than other regions. Rhyme density increased by 34.2% across all regions, with Midwest artists exhibiting the highest technical complexity (3.04 rhymes per line). Topic modeling identified significant shifts in thematic content, with social justice themes decreasing from 28.5% to 13.8% of content while introspective themes increased from 7.6% to 26.3%. Sentiment analysis demon- strated that lyrics became significantly more negative during sociopolitical crises, with polarity decreasing by 0.31 following major social unrest. Multi-dimensional analysis revealed four dis- tinct stylistic approaches that correlate strongly with geographic origin (r=0.68, p!0.001) and time period (r=0.59, p<0.001). These findings establish quantitative evidence for the evolution of hip- hop as both an art form and a reflection of societal dynamics, providing insights into the interplay between linguistic innovation and cultural context in popular music. 

---
# Improving Phishing Email Detection Performance of Small Large Language Models 

**Authors**: Zijie Lin, Zikang Liu, Hanbo Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00034)  

**Abstract**: Large language models(LLMs) have demonstrated remarkable performance on many natural language processing(NLP) tasks and have been employed in phishing email detection research. However, in current studies, well-performing LLMs typically contain billions or even tens of billions of parameters, requiring enormous computational resources. To reduce computational costs, we investigated the effectiveness of small-parameter LLMs for phishing email detection. These LLMs have around 3 billion parameters and can run on consumer-grade GPUs. However, small LLMs often perform poorly in phishing email detection task. To address these issues, we designed a set of methods including Prompt Engineering, Explanation Augmented Fine-tuning, and Model Ensemble to improve phishing email detection capabilities of small LLMs. We validated the effectiveness of our approach through experiments, significantly improving accuracy on the SpamAssassin dataset from around 0.5 for baseline models like Qwen2.5-1.5B-Instruct to 0.976. 

---
# From Attention to Atoms: Spectral Dictionary Learning for Fast, Interpretable Language Models 

**Authors**: Andrew Kiruluta  

**Link**: [PDF](https://arxiv.org/pdf/2505.00033)  

**Abstract**: We propose a novel spectral generative modeling framework for natural language processing that jointly learns a global time varying Fourier dictionary and per token mixing coefficients, replacing the ubiquitous self attention mechanism in transformer architectures. By enforcing reconstruction losses in both the time domain (embedding reconstruction) and the frequency domain (via Short Time Fourier Transform magnitude matching) alongside a standard language modeling objective, and fitting a Gaussian Mixture Model (GMM) prior over the learned mixing vectors, our approach achieves competitive perplexity and generation quality on standard benchmarks such as WikiText2 and Penn Treebank. In contrast to the quadratic computation complexity of self attention, our method operates with linear complexity, delivering substantial efficiency gains. We demonstrate that spectral dictionary models can achieve competitive performance compared to transformer baselines while significantly reducing inference latency and memory footprint, offering a compelling alternative for scalable language modeling. 

---
# MDD-LLM: Towards Accuracy Large Language Models for Major Depressive Disorder Diagnosis 

**Authors**: Yuyang Sha, Hongxin Pan, Wei Xu, Weiyu Meng, Gang Luo, Xinyu Du, Xiaobing Zhai, Henry H. Y. Tong, Caijuan Shi, Kefeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.00032)  

**Abstract**: Major depressive disorder (MDD) impacts more than 300 million people worldwide, highlighting a significant public health issue. However, the uneven distribution of medical resources and the complexity of diagnostic methods have resulted in inadequate attention to this disorder in numerous countries and regions. This paper introduces a high-performance MDD diagnosis tool named MDD-LLM, an AI-driven framework that utilizes fine-tuned large language models (LLMs) and extensive real-world samples to tackle challenges in MDD diagnosis. Therefore, we select 274,348 individual information from the UK Biobank cohort to train and evaluate the proposed method. Specifically, we select 274,348 individual records from the UK Biobank cohort and design a tabular data transformation method to create a large corpus for training and evaluating the proposed approach. To illustrate the advantages of MDD-LLM, we perform comprehensive experiments and provide several comparative analyses against existing model-based solutions across multiple evaluation metrics. Experimental results show that MDD-LLM (70B) achieves an accuracy of 0.8378 and an AUC of 0.8919 (95% CI: 0.8799 - 0.9040), significantly outperforming existing machine learning and deep learning frameworks for MDD diagnosis. Given the limited exploration of LLMs in MDD diagnosis, we examine numerous factors that may influence the performance of our proposed method, such as tabular data transformation techniques and different fine-tuning strategies. 

---
# Learning to Plan Before Answering: Self-Teaching LLMs to Learn Abstract Plans for Problem Solving 

**Authors**: Jin Zhang, Flood Sung, Zhilin Yang, Yang Gao, Chongjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00031)  

**Abstract**: In the field of large language model (LLM) post-training, the effectiveness of utilizing synthetic data generated by the LLM itself has been well-presented. However, a key question remains unaddressed: what essential information should such self-generated data encapsulate? Existing approaches only produce step-by-step problem solutions, and fail to capture the abstract meta-knowledge necessary for generalization across similar problems. Drawing insights from cognitive science, where humans employ high-level abstraction to simplify complex problems before delving into specifics, we introduce a novel self-training algorithm: LEarning to Plan before Answering (LEPA). LEPA trains the LLM to formulate anticipatory plans, which serve as abstract meta-knowledge for problem-solving, before engaging with the intricacies of problems. This approach not only outlines the solution generation path but also shields the LLM from the distraction of irrelevant details. During data generation, LEPA first crafts an anticipatory plan based on the problem, and then generates a solution that aligns with both the plan and the problem. LEPA refines the plan through self-reflection, aiming to acquire plans that are instrumental in yielding correct solutions. During model optimization, the LLM is trained to predict both the refined plans and the corresponding solutions. By efficiently extracting and utilizing the anticipatory plans, LEPA demonstrates remarkable superiority over conventional algorithms on various challenging natural language reasoning benchmarks. 

---
# Can Language Models Represent the Past without Anachronism? 

**Authors**: Ted Underwood, Laura K. Nelson, Matthew Wilkens  

**Link**: [PDF](https://arxiv.org/pdf/2505.00030)  

**Abstract**: Before researchers can use language models to simulate the past, they need to understand the risk of anachronism. We find that prompting a contemporary model with examples of period prose does not produce output consistent with period style. Fine-tuning produces results that are stylistically convincing enough to fool an automated judge, but human evaluators can still distinguish fine-tuned model outputs from authentic historical text. We tentatively conclude that pretraining on period prose may be required in order to reliably simulate historical perspectives for social research. 

---
# Keep the General, Inject the Specific: Structured Dialogue Fine-Tuning for Knowledge Injection without Catastrophic Forgetting 

**Authors**: Yijie Hong, Xiaofei Yin, Xinzhong Wang, Yi Tu, Ya Guo, Sufeng Duan, Weiqiang Wang, Lingyong Fang, Depeng Wang, Huijia Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00029)  

**Abstract**: Large Vision Language Models have demonstrated impressive versatile capabilities through extensive multimodal pre-training, but face significant limitations when incorporating specialized knowledge domains beyond their training distribution. These models struggle with a fundamental dilemma: direct adaptation approaches that inject domain-specific knowledge often trigger catastrophic forgetting of foundational visual-linguistic abilities. We introduce Structured Dialogue Fine-Tuning (SDFT), an effective approach that effectively injects domain-specific knowledge while minimizing catastrophic forgetting. Drawing inspiration from supervised fine-tuning in LLMs and subject-driven personalization in text-to-image diffusion models, our method employs a three-phase dialogue structure: Foundation Preservation reinforces pre-trained visual-linguistic alignment through caption tasks; Contrastive Disambiguation introduces carefully designed counterfactual examples to maintain semantic boundaries; and Knowledge Specialization embeds specialized information through chain-of-thought reasoning. Experimental results across multiple domains confirm SDFT's effectiveness in balancing specialized knowledge acquisition with general capability retention. Our key contributions include a data-centric dialogue template that balances foundational alignment with targeted knowledge integration, a weighted multi-turn supervision framework, and comprehensive evaluation across diverse knowledge types. 

---
# Enhancing Speech-to-Speech Dialogue Modeling with End-to-End Retrieval-Augmented Generation 

**Authors**: Pengchao Feng, Ziyang Ma, Wenxi Chen, Yao Li, Sheng Wang, Kai Yu, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00028)  

**Abstract**: In recent years, end-to-end speech-to-speech (S2S) dialogue systems have garnered increasing research attention due to their advantages over traditional cascaded systems, including achieving lower latency and more natural integration of nonverbal cues such as emotion and speaker identity. However, these end-to-end systems face key challenges, particularly in incorporating external knowledge, a capability commonly addressed by Retrieval-Augmented Generation (RAG) in text-based large language models (LLMs). The core difficulty lies in the modality gap between input speech and retrieved textual knowledge, which hinders effective integration. To address this issue, we propose a novel end-to-end RAG framework that directly retrieves relevant textual knowledge from speech queries, eliminating the need for intermediate speech-to-text conversion via techniques like ASR. Experimental results demonstrate that our method significantly improves the performance of end-to-end S2S dialogue systems while achieving higher retrieval efficiency. Although the overall performance still lags behind cascaded models, our framework offers a promising direction for enhancing knowledge integration in end-to-end S2S systems. We will release the code and dataset to support reproducibility and promote further research in this area. 

---
# Extracting Abstraction Dimensions by Identifying Syntax Pattern from Texts 

**Authors**: Jian Zhou, Jiazheng Li, Sirui Zhuge, Hai Zhuge  

**Link**: [PDF](https://arxiv.org/pdf/2505.00027)  

**Abstract**: This paper proposed an approach to automatically discovering subject dimension, action dimension, object dimension and adverbial dimension from texts to efficiently operate texts and support query in natural language. The high quality of trees guarantees that all subjects, actions, objects and adverbials and their subclass relations within texts can be represented. The independency of trees ensures that there is no redundant representation between trees. The expressiveness of trees ensures that the majority of sentences can be accessed from each tree and the rest of sentences can be accessed from at least one tree so that the tree-based search mechanism can support querying in natural language. Experiments show that the average precision, recall and F1-score of the abstraction trees constructed by the subclass relations of subject, action, object and adverbial are all greater than 80%. The application of the proposed approach to supporting query in natural language demonstrates that different types of question patterns for querying subject or object have high coverage of texts, and searching multiple trees on subject, action, object and adverbial according to the question pattern can quickly reduce search space to locate target sentences, which can support precise operation on texts. 

---
# Theory of Mind in Large Language Models: Assessment and Enhancement 

**Authors**: Ruirui Chen, Weifeng Jiang, Chengwei Qin, Cheston Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00026)  

**Abstract**: Theory of Mind (ToM)-the ability to infer and reason about others' mental states-is fundamental to human social intelligence. As Large Language Models (LLMs) become increasingly integrated into daily life, it is crucial to assess and enhance their capacity to interpret and respond to human mental states. In this paper, we review LLMs' ToM capabilities by examining both evaluation benchmarks and the strategies designed to improve them. We focus on widely adopted story-based benchmarks and provide an in-depth analysis of methods aimed at enhancing ToM in LLMs. Furthermore, we outline promising future research directions informed by recent benchmarks and state-of-the-art approaches. Our survey serves as a valuable resource for researchers interested in advancing LLMs' ToM capabilities. 

---
# A Method for the Architecture of a Medical Vertical Large Language Model Based on Deepseek R1 

**Authors**: Mingda Zhang, Jianglong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2505.00025)  

**Abstract**: In recent years, despite foundation models like DeepSeek-R1 and ChatGPT demonstrating significant capabilities in general tasks, professional knowledge barriers, computational resource requirements, and deployment environment limitations have severely hindered their application in actual medical scenarios. Addressing these challenges, this paper proposes an efficient lightweight medical vertical large language model architecture method, systematically solving the lightweight problem of medical large models from three dimensions: knowledge acquisition, model compression, and computational optimization. At the knowledge acquisition level, a knowledge transfer pipeline is designed from the fine-tuned DeepSeek-R1-Distill-70B teacher model to the DeepSeek-R1-Distill-7B student model, and Low-Rank Adaptation (LoRA) technology is adopted to precisely adjust key attention layers. At the model compression level, compression techniques including 4-bit weight quantization are implemented while preserving the core representation ability for medical reasoning. At the computational optimization level, inference optimization techniques such as Flash Attention acceleration and continuous batching are integrated, and a professional prompt template system is constructed to adapt to different types of medical problems. Experimental results on medical question-answering datasets show that the method proposed in this paper maintains professional accuracy while reducing memory consumption by 64.7\% and inference latency by 12.4\%, providing an effective solution for the application of medical large models in resource-constrained environments such as edge computing devices. 

---
# Nemotron-Research-Tool-N1: Tool-Using Language Models with Reinforced Reasoning 

**Authors**: Shaokun Zhang, Yi Dong, Jieyu Zhang, Jan Kautz, Bryan Catanzaro, Andrew Tao, Qingyun Wu, Zhiding Yu, Guilin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00024)  

**Abstract**: Enabling large language models with external tools has become a pivotal strategy for extending their functionality beyond text generation tasks. Prior work typically enhances tool-use abilities by either applying supervised fine-tuning (SFT) to enforce tool-call correctness or distilling reasoning traces from stronger models for SFT. However, both approaches fall short, either omitting reasoning entirely or producing imitative reasoning that limits generalization. Inspired by the success of DeepSeek-R1 in eliciting reasoning through rule-based reinforcement learning, we develop the Nemotron-Research-Tool-N1 series of tool-using language models using a similar training paradigm. Instead of restrictively supervising intermediate reasoning traces distilled from stronger models, Nemotron-Research-Tool-N1 is optimized with a binary reward that evaluates only the structural validity and functional correctness of tool invocations. This lightweight supervision allows the model to autonomously internalize reasoning strategies, without the need for annotated reasoning trajectories. Experiments on the BFCL and API-Bank benchmarks show that Nemotron-Research-Tool-N1-7B and Nemotron-Research-Tool-N1-14B, built on Qwen-2.5-7B/14B-Instruct, achieve state-of-the-art results, outperforming GPT-4o on both evaluations. 

---
# CORG: Generating Answers from Complex, Interrelated Contexts 

**Authors**: Hyunji Lee, Franck Dernoncourt, Trung Bui, Seunghyun Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2505.00023)  

**Abstract**: In a real-world corpus, knowledge frequently recurs across documents but often contains inconsistencies due to ambiguous naming, outdated information, or errors, leading to complex interrelationships between contexts. Previous research has shown that language models struggle with these complexities, typically focusing on single factors in isolation. We classify these relationships into four types: distracting, ambiguous, counterfactual, and duplicated. Our analysis reveals that no single approach effectively addresses all these interrelationships simultaneously. Therefore, we introduce Context Organizer (CORG), a framework that organizes multiple contexts into independently processed groups. This design allows the model to efficiently find all relevant answers while ensuring disambiguation. CORG consists of three key components: a graph constructor, a reranker, and an aggregator. Our results demonstrate that CORG balances performance and efficiency effectively, outperforming existing grouping methods and achieving comparable results to more computationally intensive, single-context approaches. 

---
# Aleph-Alpha-GermanWeb: Improving German-language LLM pre-training with model-based data curation and synthetic data generation 

**Authors**: Thomas F Burns, Letitia Parcalabescu, Stephan Wäldchen, Michael Barlow, Gregor Ziegltrum, Volker Stampa, Bastian Harren, Björn Deiseroth  

**Link**: [PDF](https://arxiv.org/pdf/2505.00022)  

**Abstract**: Scaling data quantity is essential for large language models (LLMs), yet recent findings show that data quality can significantly boost performance and training efficiency. We introduce a German-language dataset curation pipeline that combines heuristic and model-based filtering techniques with synthetic data generation. We use our pipeline to create Aleph-Alpha-GermanWeb, a large-scale German pre-training dataset which draws from: (1) Common Crawl web data, (2) FineWeb2, and (3) synthetically-generated data conditioned on actual, organic web data. We evaluate our dataset by pre-training both a 1B Llama-style model and an 8B tokenizer-free hierarchical autoregressive transformer (HAT). A comparison on German-language benchmarks, including MMMLU, shows significant performance gains of Aleph-Alpha-GermanWeb over FineWeb2 alone. This advantage holds at the 8B scale even when FineWeb2 is enriched by human-curated high-quality data sources such as Wikipedia. Our findings support the growing body of evidence that model-based data curation and synthetic data generation can significantly enhance LLM pre-training datasets. 

---
# Ustnlp16 at SemEval-2025 Task 9: Improving Model Performance through Imbalance Handling and Focal Loss 

**Authors**: Zhuoang Cai, Zhenghao Li, Yang Liu, Liyuan Guo, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.00021)  

**Abstract**: Classification tasks often suffer from imbal- anced data distribution, which presents chal- lenges in food hazard detection due to severe class imbalances, short and unstructured text, and overlapping semantic categories. In this paper, we present our system for SemEval- 2025 Task 9: Food Hazard Detection, which ad- dresses these issues by applying data augmenta- tion techniques to improve classification perfor- mance. We utilize transformer-based models, BERT and RoBERTa, as backbone classifiers and explore various data balancing strategies, including random oversampling, Easy Data Augmentation (EDA), and focal loss. Our ex- periments show that EDA effectively mitigates class imbalance, leading to significant improve- ments in accuracy and F1 scores. Furthermore, combining focal loss with oversampling and EDA further enhances model robustness, par- ticularly for hard-to-classify examples. These findings contribute to the development of more effective NLP-based classification models for food hazard detection. 

---
# Beyond Public Access in LLM Pre-Training Data 

**Authors**: Sruly Rosenblat, Tim O'Reilly, Ilan Strauss  

**Link**: [PDF](https://arxiv.org/pdf/2505.00020)  

**Abstract**: Using a legally obtained dataset of 34 copyrighted O'Reilly Media books, we apply the DE-COP membership inference attack method to investigate whether OpenAI's large language models were trained on copyrighted content without consent. Our AUROC scores show that GPT-4o, OpenAI's more recent and capable model, demonstrates strong recognition of paywalled O'Reilly book content (AUROC = 82\%), compared to OpenAI's earlier model GPT-3.5 Turbo. In contrast, GPT-3.5 Turbo shows greater relative recognition of publicly accessible O'Reilly book samples. GPT-4o Mini, as a much smaller model, shows no knowledge of public or non-public O'Reilly Media content when tested (AUROC $\approx$ 50\%). Testing multiple models, with the same cutoff date, helps us account for potential language shifts over time that might bias our findings. These results highlight the urgent need for increased corporate transparency regarding pre-training data sources as a means to develop formal licensing frameworks for AI content training 

---
# An Empirical Study on Prompt Compression for Large Language Models 

**Authors**: Zheng Zhang, Jinyi Li, Yihuai Lan, Xiang Wang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00019)  

**Abstract**: Prompt engineering enables Large Language Models (LLMs) to perform a variety of tasks. However, lengthy prompts significantly increase computational complexity and economic costs. To address this issue, we study six prompt compression methods for LLMs, aiming to reduce prompt length while maintaining LLM response quality. In this paper, we present a comprehensive analysis covering aspects such as generation performance, model hallucinations, efficacy in multimodal tasks, word omission analysis, and more. We evaluate these methods across 13 datasets, including news, scientific articles, commonsense QA, math QA, long-context QA, and VQA datasets. Our experiments reveal that prompt compression has a greater impact on LLM performance in long contexts compared to short ones. In the Longbench evaluation, moderate compression even enhances LLM performance. Our code and data is available at this https URL. 

---
# ReCellTy: Domain-specific knowledge graph retrieval-augmented LLMs workflow for single-cell annotation 

**Authors**: Dezheng Han, Yibin Jia, Ruxiao Chen, Wenjie Han, Shuaishuai Guo, Jianbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00017)  

**Abstract**: To enable precise and fully automated cell type annotation with large language models (LLMs), we developed a graph structured feature marker database to retrieve entities linked to differential genes for cell reconstruction. We further designed a multi task workflow to optimize the annotation process. Compared to general purpose LLMs, our method improves human evaluation scores by up to 0.21 and semantic similarity by 6.1% across 11 tissue types, while more closely aligning with the cognitive logic of manual annotation. 

---
# Sparks of Tabular Reasoning via Text2SQL Reinforcement Learning 

**Authors**: Josefa Lia Stoisser, Marc Boubnovski Martell, Julien Fauqueur  

**Link**: [PDF](https://arxiv.org/pdf/2505.00016)  

**Abstract**: This work reframes the Text-to-SQL task as a pathway for teaching large language models (LLMs) to reason over and manipulate tabular data--moving beyond the traditional focus on query generation. We propose a two-stage framework that leverages SQL supervision to develop transferable table reasoning capabilities. First, we synthesize detailed chain-of-thought (CoT) traces from real-world SQL queries, providing step-by-step, clause-level supervision that teaches the model how to traverse, filter, and aggregate table fields. Second, we introduce a Group Relative Policy Optimization (GRPO) reinforcement learning objective that connects SQL execution accuracy to generalizable reasoning by encouraging steps that extend beyond task-specific syntax and transfer across datasets. Empirically, our approach improves performance on standard Text-to-SQL benchmarks and achieves substantial gains on reasoning-intensive datasets such as BIRD and CRT-QA, demonstrating enhanced generalization and interpretability. Specifically, the distilled-quantized LLaMA model achieved a 20\% increase in accuracy when trained on Text-to-SQL tasks, while Qwen achieved a 5\% increase. These results suggest that SQL can serve not only as a target formalism but also as an effective scaffold for learning robust, transferable reasoning over structured data. 

---
# Design and Application of Multimodal Large Language Model Based System for End to End Automation of Accident Dataset Generation 

**Authors**: MD Thamed Bin Zaman Chowdhury, Moazzem Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2505.00015)  

**Abstract**: Road traffic accidents remain a major public safety and socio-economic issue in developing countries like Bangladesh. Existing accident data collection is largely manual, fragmented, and unreliable, resulting in underreporting and inconsistent records. This research proposes a fully automated system using Large Language Models (LLMs) and web scraping techniques to address these challenges. The pipeline consists of four components: automated web scraping code generation, news collection from online sources, accident news classification with structured data extraction, and duplicate removal. The system uses the multimodal generative LLM Gemini-2.0-Flash for seamless automation. The code generation module classifies webpages into pagination, dynamic, or infinite scrolling categories and generates suitable Python scripts for scraping. LLMs also classify and extract key accident information such as date, time, location, fatalities, injuries, road type, vehicle types, and pedestrian involvement. A deduplication algorithm ensures data integrity by removing duplicate reports. The system scraped 14 major Bangladeshi news sites over 111 days (Oct 1, 2024 - Jan 20, 2025), processing over 15,000 news articles and identifying 705 unique accidents. The code generation module achieved 91.3% calibration and 80% validation accuracy. Chittagong reported the highest number of accidents (80), fatalities (70), and injuries (115), followed by Dhaka, Faridpur, Gazipur, and Cox's Bazar. Peak accident times were morning (8-9 AM), noon (12-1 PM), and evening (6-7 PM). A public repository was also developed with usage instructions. This study demonstrates the viability of an LLM-powered, scalable system for accurate, low-effort accident data collection, providing a foundation for data-driven road safety policymaking in Bangladesh. 

---
# Manifold-Constrained Sentence Embeddings via Triplet Loss: Projecting Semantics onto Spheres, Tori, and Möbius Strips 

**Authors**: Vinit K. Chavan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00014)  

**Abstract**: Recent advances in representation learning have emphasized the role of embedding geometry in capturing semantic structure. Traditional sentence embeddings typically reside in unconstrained Euclidean spaces, which may limit their ability to reflect complex relationships in language. In this work, we introduce a novel framework that constrains sentence embeddings to lie on continuous manifolds -- specifically the unit sphere, torus, and Möbius strip -- using triplet loss as the core training objective. By enforcing differential geometric constraints on the output space, our approach encourages the learning of embeddings that are both discriminative and topologically structured.
We evaluate our method on benchmark datasets (AG News and MBTI) and compare it to classical baselines including TF-IDF, Word2Vec, and unconstrained Keras-derived embeddings. Our results demonstrate that manifold-constrained embeddings, particularly those projected onto spheres and Möbius strips, significantly outperform traditional approaches in both clustering quality (Silhouette Score) and classification performance (Accuracy). These findings highlight the value of embedding in manifold space -- where topological structure complements semantic separation -- offering a new and mathematically grounded direction for geometric representation learning in NLP. 

---
# Performance Evaluation of Emotion Classification in Japanese Using RoBERTa and DeBERTa 

**Authors**: Yoichi Takenaka  

**Link**: [PDF](https://arxiv.org/pdf/2505.00013)  

**Abstract**: Background Practical applications such as social media monitoring and customer-feedback analysis require accurate emotion detection for Japanese text, yet resource scarcity and class imbalance hinder model performance.
Objective This study aims to build a high-accuracy model for predicting the presence or absence of eight Plutchik emotions in Japanese sentences.
Methods Using the WRIME corpus, we transform reader-averaged intensity scores into binary labels and fine-tune four pre-trained language models (BERT, RoBERTa, DeBERTa-v3-base, DeBERTa-v3-large). For context, we also assess two large language models (TinySwallow-1.5B-Instruct and ChatGPT-4o). Accuracy and F1-score serve as evaluation metrics.
Results DeBERTa-v3-large attains the best mean accuracy (0.860) and F1-score (0.662), outperforming all other models. It maintains robust F1 across both high-frequency emotions (e.g., Joy, Anticipation) and low-frequency emotions (e.g., Anger, Trust). The LLMs lag, with ChatGPT-4o and TinySwallow-1.5B-Instruct scoring 0.527 and 0.292 in mean F1, respectively.
Conclusion The fine-tuned DeBERTa-v3-large model currently offers the most reliable solution for binary emotion classification in Japanese. We release this model as a pip-installable package (pip install deberta-emotion-predictor). Future work should augment data for rare emotions, reduce model size, and explore prompt engineering to improve LLM performance.
This manuscript is under review for possible publication in New Generation Computing. 

---
# The AI Co-Ethnographer: How Far Can Automation Take Qualitative Research? 

**Authors**: Fabian Retkowski, Andreas Sudmann, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2505.00012)  

**Abstract**: Qualitative research often involves labor-intensive processes that are difficult to scale while preserving analytical depth. This paper introduces The AI Co-Ethnographer (AICoE), a novel end-to-end pipeline developed for qualitative research and designed to move beyond the limitations of simply automating code assignments, offering a more integrated approach. AICoE organizes the entire process, encompassing open coding, code consolidation, code application, and even pattern discovery, leading to a comprehensive analysis of qualitative data. 

---
# Jailbreak Detection in Clinical Training LLMs Using Feature-Based Predictive Models 

**Authors**: Tri Nguyen, Lohith Srikanth Pentapalli, Magnus Sieverding, Laurah Turner, Seth Overla, Weibing Zheng, Chris Zhou, David Furniss, Danielle Weber, Michael Gharib, Matt Kelleher, Michael Shukis, Cameron Pawlik, Kelly Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00010)  

**Abstract**: Jailbreaking in Large Language Models (LLMs) threatens their safe use in sensitive domains like education by allowing users to bypass ethical safeguards. This study focuses on detecting jailbreaks in 2-Sigma, a clinical education platform that simulates patient interactions using LLMs. We annotated over 2,300 prompts across 158 conversations using four linguistic variables shown to correlate strongly with jailbreak behavior. The extracted features were used to train several predictive models, including Decision Trees, Fuzzy Logic-based classifiers, Boosting methods, and Logistic Regression. Results show that feature-based predictive models consistently outperformed Prompt Engineering, with the Fuzzy Decision Tree achieving the best overall performance. Our findings demonstrate that linguistic-feature-based models are effective and explainable alternatives for jailbreak detection. We suggest future work explore hybrid frameworks that integrate prompt-based flexibility with rule-based robustness for real-time, spectrum-based jailbreak monitoring in educational LLMs. 

---
# Efficient Knowledge Transfer in Multi-Task Learning through Task-Adaptive Low-Rank Representation 

**Authors**: Xiao Zhang, Kangsheng Wang, Tianyu Hu, Huimin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.00009)  

**Abstract**: Pre-trained language models (PLMs) demonstrate remarkable intelligence but struggle with emerging tasks unseen during training in real-world applications. Training separate models for each new task is usually impractical. Multi-task learning (MTL) addresses this challenge by transferring shared knowledge from source tasks to target tasks. As an dominant parameter-efficient fine-tuning method, prompt tuning (PT) enhances MTL by introducing an adaptable vector that captures task-specific knowledge, which acts as a prefix to the original prompt that preserves shared knowledge, while keeping PLM parameters frozen. However, PT struggles to effectively capture the heterogeneity of task-specific knowledge due to its limited representational capacity. To address this challenge, we propose Task-Adaptive Low-Rank Representation (TA-LoRA), an MTL method built on PT, employing the low-rank representation to model task heterogeneity and a fast-slow weights mechanism where the slow weight encodes shared knowledge, while the fast weight captures task-specific nuances, avoiding the mixing of shared and task-specific knowledge, caused by training low-rank representations from scratch. Moreover, a zero-initialized attention mechanism is introduced to minimize the disruption of immature low-rank components on original prompts during warm-up epochs. Experiments on 16 tasks demonstrate that TA-LoRA achieves state-of-the-art performance in full-data and few-shot settings while maintaining superior parameter efficiency. 

---
# A Scoping Review of Natural Language Processing in Addressing Medically Inaccurate Information: Errors, Misinformation, and Hallucination 

**Authors**: Zhaoyi Sun, Wen-Wai Yim, Ozlem Uzuner, Fei Xia, Meliha Yetisgen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00008)  

**Abstract**: Objective: This review aims to explore the potential and challenges of using Natural Language Processing (NLP) to detect, correct, and mitigate medically inaccurate information, including errors, misinformation, and hallucination. By unifying these concepts, the review emphasizes their shared methodological foundations and their distinct implications for healthcare. Our goal is to advance patient safety, improve public health communication, and support the development of more reliable and transparent NLP applications in healthcare.
Methods: A scoping review was conducted following PRISMA guidelines, analyzing studies from 2020 to 2024 across five databases. Studies were selected based on their use of NLP to address medically inaccurate information and were categorized by topic, tasks, document types, datasets, models, and evaluation metrics.
Results: NLP has shown potential in addressing medically inaccurate information on the following tasks: (1) error detection (2) error correction (3) misinformation detection (4) misinformation correction (5) hallucination detection (6) hallucination mitigation. However, challenges remain with data privacy, context dependency, and evaluation standards.
Conclusion: This review highlights the advancements in applying NLP to tackle medically inaccurate information while underscoring the need to address persistent challenges. Future efforts should focus on developing real-world datasets, refining contextual methods, and improving hallucination management to ensure reliable and transparent healthcare applications. 

---
# Toward a digital twin of U.S. Congress 

**Authors**: Hayden Helm, Tianyi Chen, Harvey McGuinness, Paige Lee, Brandon Duderstadt, Carey E. Priebe  

**Link**: [PDF](https://arxiv.org/pdf/2505.00006)  

**Abstract**: In this paper we provide evidence that a virtual model of U.S. congresspersons based on a collection of language models satisfies the definition of a digital twin. In particular, we introduce and provide high-level descriptions of a daily-updated dataset that contains every Tweet from every U.S. congressperson during their respective terms. We demonstrate that a modern language model equipped with congressperson-specific subsets of this data are capable of producing Tweets that are largely indistinguishable from actual Tweets posted by their physical counterparts. We illustrate how generated Tweets can be used to predict roll-call vote behaviors and to quantify the likelihood of congresspersons crossing party lines, thereby assisting stakeholders in allocating resources and potentially impacting real-world legislative dynamics. We conclude with a discussion of the limitations and important extensions of our analysis. 

---
# LangVAE and LangSpace: Building and Probing for Language Model VAEs 

**Authors**: Danilo S. Carvalho, Yingji Zhang, Harriet Unsworth, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2505.00004)  

**Abstract**: We present LangVAE, a novel framework for modular construction of variational autoencoders (VAEs) on top of pre-trained large language models (LLMs). Such language model VAEs can encode the knowledge of their pre-trained components into more compact and semantically disentangled representations. The representations obtained in this way can be analysed with the LangVAE companion framework: LangSpace, which implements a collection of probing methods, such as vector traversal and interpolation, disentanglement measures, and cluster visualisations. LangVAE and LangSpace offer a flexible, efficient and scalable way of building and analysing textual representations, with simple integration for models available on the HuggingFace Hub. Additionally, we conducted a set of experiments with different encoder and decoder combinations, as well as annotated inputs, revealing a wide range of interactions across architectural families and sizes w.r.t. generalisation and disentanglement. Our findings demonstrate a promising framework for systematising the experimentation and understanding of textual representations. 

---
# The Mind in the Machine: A Survey of Incorporating Psychological Theories in LLMs 

**Authors**: Zizhou Liu, Ziwei Gong, Lin Ai, Zheng Hui, Run Chen, Colin Wayne Leach, Michelle R. Greene, Julia Hirschberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.00003)  

**Abstract**: Psychological insights have long shaped pivotal NLP breakthroughs, including the cognitive underpinnings of attention mechanisms, formative reinforcement learning, and Theory of Mind-inspired social modeling. As Large Language Models (LLMs) continue to grow in scale and complexity, there is a rising consensus that psychology is essential for capturing human-like cognition, behavior, and interaction. This paper reviews how psychological theories can inform and enhance stages of LLM development, including data, pre-training, post-training, and evaluation\&application. Our survey integrates insights from cognitive, developmental, behavioral, social, personality psychology, and psycholinguistics. Our analysis highlights current trends and gaps in how psychological theories are applied. By examining both cross-domain connections and points of tension, we aim to bridge disciplinary divides and promote more thoughtful integration of psychology into future NLP research. 

---
# Symbol grounding in computational systems: A paradox of intentions 

**Authors**: Vincent C. Müller  

**Link**: [PDF](https://arxiv.org/pdf/2505.00002)  

**Abstract**: The paper presents a paradoxical feature of computational systems that suggests that computationalism cannot explain symbol grounding. If the mind is a digital computer, as computationalism claims, then it can be computing either over meaningful symbols or over meaningless symbols. If it is computing over meaningful symbols its functioning presupposes the existence of meaningful symbols in the system, i.e. it implies semantic nativism. If the mind is computing over meaningless symbols, no intentional cognitive processes are available prior to symbol grounding. In this case, no symbol grounding could take place since any grounding presupposes intentional cognitive processes. So, whether computing in the mind is over meaningless or over meaningful symbols, computationalism implies semantic nativism. 

---
# Rosetta-PL: Propositional Logic as a Benchmark for Large Language Model Reasoning 

**Authors**: Shaun Baek, Shaun Esua-Mensah, Cyrus Tsui, Sejan Vigneswaralingam, Abdullah Alali, Michael Lu, Vasu Sharma, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00001)  

**Abstract**: Large Language Models (LLMs) are primarily trained on high-resource natural languages, limiting their effectiveness in low-resource settings and in tasks requiring deep logical reasoning. This research introduces Rosetta-PL, a benchmark designed to evaluate LLMs' logical reasoning and generalization capabilities in a controlled environment. We construct Rosetta-PL by translating a dataset of logical propositions from Lean into a custom logical language, which is then used to fine-tune an LLM (e.g., GPT-4o). Our experiments analyze the impact of the size of the dataset and the translation methodology on the performance of the model. Our results indicate that preserving logical relationships in the translation process significantly boosts precision, with accuracy plateauing beyond roughly 20,000 training samples. These insights provide valuable guidelines for optimizing LLM training in formal reasoning tasks and improving performance in various low-resource language applications. 

---
# T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT 

**Authors**: Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, Pheng-Ann Heng, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.00703)  

**Abstract**: Recent advancements in large language models have demonstrated how chain-of-thought (CoT) and reinforcement learning (RL) can improve performance. However, applying such reasoning strategies to the visual generation domain remains largely unexplored. In this paper, we present T2I-R1, a novel reasoning-enhanced text-to-image generation model, powered by RL with a bi-level CoT reasoning process. Specifically, we identify two levels of CoT that can be utilized to enhance different stages of generation: (1) the semantic-level CoT for high-level planning of the prompt and (2) the token-level CoT for low-level pixel processing during patch-by-patch generation. To better coordinate these two levels of CoT, we introduce BiCoT-GRPO with an ensemble of generation rewards, which seamlessly optimizes both generation CoTs within the same training step. By applying our reasoning strategies to the baseline model, Janus-Pro, we achieve superior performance with 13% improvement on T2I-CompBench and 19% improvement on the WISE benchmark, even surpassing the state-of-the-art model FLUX.1. Code is available at: this https URL 

---
# Investigating Task Arithmetic for Zero-Shot Information Retrieval 

**Authors**: Marco Braga, Pranav Kasela, Alessandro Raganato, Gabriella Pasi  

**Link**: [PDF](https://arxiv.org/pdf/2505.00649)  

**Abstract**: Large Language Models (LLMs) have shown impressive zero-shot performance across a variety of Natural Language Processing tasks, including document re-ranking. However, their effectiveness degrades on unseen tasks and domains, largely due to shifts in vocabulary and word distributions. In this paper, we investigate Task Arithmetic, a technique that combines the weights of LLMs pre-trained on different tasks or domains via simple mathematical operations, such as addition or subtraction, to adapt retrieval models without requiring additional fine-tuning. Our method is able to synthesize diverse tasks and domain knowledge into a single model, enabling effective zero-shot adaptation in different retrieval contexts. Extensive experiments on publicly available scientific, biomedical, and multilingual datasets show that our method improves state-of-the-art re-ranking performance by up to 18% in NDCG@10 and 15% in P@10. In addition to these empirical gains, our analysis provides insights into the strengths and limitations of Task Arithmetic as a practical strategy for zero-shot learning and model adaptation. We make our code publicly available at this https URL. 

---
# Toward Automated Regulatory Decision-Making: Trustworthy Medical Device Risk Classification with Multimodal Transformers and Self-Training 

**Authors**: Yu Han, Aaron Ceross, Jeroen H.M. Bergmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.00422)  

**Abstract**: Accurate classification of medical device risk levels is essential for regulatory oversight and clinical safety. We present a Transformer-based multimodal framework that integrates textual descriptions and visual information to predict device regulatory classification. The model incorporates a cross-attention mechanism to capture intermodal dependencies and employs a self-training strategy for improved generalization under limited supervision. Experiments on a real-world regulatory dataset demonstrate that our approach achieves up to 90.4% accuracy and 97.9% AUROC, significantly outperforming text-only (77.2%) and image-only (54.8%) baselines. Compared to standard multimodal fusion, the self-training mechanism improved SVM performance by 3.3 percentage points in accuracy (from 87.1% to 90.4%) and 1.4 points in macro-F1, suggesting that pseudo-labeling can effectively enhance generalization under limited supervision. Ablation studies further confirm the complementary benefits of both cross-modal attention and self-training. 

---
# R&B: Domain Regrouping and Data Mixture Balancing for Efficient Foundation Model Training 

**Authors**: Albert Ge, Tzu-Heng Huang, John Cooper, Avi Trost, Ziyi Chu, Satya Sai Srinath Namburi GNVV, Ziyang Cai, Kendall Park, Nicholas Roberts, Frederic Sala  

**Link**: [PDF](https://arxiv.org/pdf/2505.00358)  

**Abstract**: Data mixing strategies have successfully reduced the costs involved in training language models. While promising, such methods suffer from two flaws. First, they rely on predetermined data domains (e.g., data sources, task types), which may fail to capture critical semantic nuances, leaving performance on the table. Second, these methods scale with the number of domains in a computationally prohibitive way. We address these challenges via R&B, a framework that re-partitions training data based on semantic similarity (Regroup) to create finer-grained domains, and efficiently optimizes the data composition (Balance) by leveraging a Gram matrix induced by domain gradients obtained throughout training. Unlike prior works, it removes the need for additional compute to obtain evaluation information such as losses or gradients. We analyze this technique under standard regularity conditions and provide theoretical insights that justify R&B's effectiveness compared to non-adaptive mixing approaches. Empirically, we demonstrate the effectiveness of R&B on five diverse datasets ranging from natural language to reasoning and multimodal tasks. With as little as 0.01% additional compute overhead, R&B matches or exceeds the performance of state-of-the-art data mixing strategies. 

---
# T2VPhysBench: A First-Principles Benchmark for Physical Consistency in Text-to-Video Generation 

**Authors**: Xuyang Guo, Jiayan Huo, Zhenmei Shi, Zhao Song, Jiahao Zhang, Jiale Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.00337)  

**Abstract**: Text-to-video generative models have made significant strides in recent years, producing high-quality videos that excel in both aesthetic appeal and accurate instruction following, and have become central to digital art creation and user engagement online. Yet, despite these advancements, their ability to respect fundamental physical laws remains largely untested: many outputs still violate basic constraints such as rigid-body collisions, energy conservation, and gravitational dynamics, resulting in unrealistic or even misleading content. Existing physical-evaluation benchmarks typically rely on automatic, pixel-level metrics applied to simplistic, life-scenario prompts, and thus overlook both human judgment and first-principles physics. To fill this gap, we introduce \textbf{T2VPhysBench}, a first-principled benchmark that systematically evaluates whether state-of-the-art text-to-video systems, both open-source and commercial, obey twelve core physical laws including Newtonian mechanics, conservation principles, and phenomenological effects. Our benchmark employs a rigorous human evaluation protocol and includes three targeted studies: (1) an overall compliance assessment showing that all models score below 0.60 on average in each law category; (2) a prompt-hint ablation revealing that even detailed, law-specific hints fail to remedy physics violations; and (3) a counterfactual robustness test demonstrating that models often generate videos that explicitly break physical rules when so instructed. The results expose persistent limitations in current architectures and offer concrete insights for guiding future research toward truly physics-aware video generation. 

---
# Mixture of Sparse Attention: Content-Based Learnable Sparse Attention via Expert-Choice Routing 

**Authors**: Piotr Piękos, Róbert Csordás, Jürgen Schmidhuber  

**Link**: [PDF](https://arxiv.org/pdf/2505.00315)  

**Abstract**: Recent advances in large language models highlighted the excessive quadratic cost of self-attention. Despite the significant research efforts, subquadratic attention methods still suffer from inferior performance in practice. We hypothesize that dynamic, learned content-based sparsity can lead to more efficient attention mechanisms. We present Mixture of Sparse Attention (MoSA), a novel approach inspired by Mixture of Experts (MoE) with expert choice routing. MoSA dynamically selects tokens for each attention head, allowing arbitrary sparse attention patterns. By selecting $k$ tokens from a sequence of length $T$, MoSA reduces the computational complexity of each attention head from $O(T^2)$ to $O(k^2 + T)$. This enables using more heads within the same computational budget, allowing higher specialization. We show that among the tested sparse attention variants, MoSA is the only one that can outperform the dense baseline, sometimes with up to 27% better perplexity for an identical compute budget. MoSA can also reduce the resource usage compared to dense self-attention. Despite using torch implementation without an optimized kernel, perplexity-matched MoSA models are simultaneously faster in wall-clock time, require less memory for training, and drastically reduce the size of the KV-cache compared to the dense transformer baselines. 

---
# EnronQA: Towards Personalized RAG over Private Documents 

**Authors**: Michael J. Ryan, Danmei Xu, Chris Nivera, Daniel Campos  

**Link**: [PDF](https://arxiv.org/pdf/2505.00263)  

**Abstract**: Retrieval Augmented Generation (RAG) has become one of the most popular methods for bringing knowledge-intensive context to large language models (LLM) because of its ability to bring local context at inference time without the cost or data leakage risks associated with fine-tuning. A clear separation of private information from the LLM training has made RAG the basis for many enterprise LLM workloads as it allows the company to augment LLM's understanding using customers' private documents. Despite its popularity for private documents in enterprise deployments, current RAG benchmarks for validating and optimizing RAG pipelines draw their corpora from public data such as Wikipedia or generic web pages and offer little to no personal context. Seeking to empower more personal and private RAG we release the EnronQA benchmark, a dataset of 103,638 emails with 528,304 question-answer pairs across 150 different user inboxes. EnronQA enables better benchmarking of RAG pipelines over private data and allows for experimentation on the introduction of personalized retrieval settings over realistic data. Finally, we use EnronQA to explore the tradeoff in memorization and retrieval when reasoning over private documents. 

---
# Self-Generated In-Context Examples Improve LLM Agents for Sequential Decision-Making Tasks 

**Authors**: Vishnu Sarukkai, Zhiqiang Xie, Kayvon Fatahalian  

**Link**: [PDF](https://arxiv.org/pdf/2505.00234)  

**Abstract**: Many methods for improving Large Language Model (LLM) agents for sequential decision-making tasks depend on task-specific knowledge engineering--such as prompt tuning, curated in-context examples, or customized observation and action spaces. Using these approaches, agent performance improves with the quality or amount of knowledge engineering invested. Instead, we investigate how LLM agents can automatically improve their performance by learning in-context from their own successful experiences on similar tasks. Rather than relying on task-specific knowledge engineering, we focus on constructing and refining a database of self-generated examples. We demonstrate that even a naive accumulation of successful trajectories across training tasks boosts test performance on three benchmarks: ALFWorld (73% to 89%), Wordcraft (55% to 64%), and InterCode-SQL (75% to 79%)--matching the performance the initial agent achieves if allowed two to three attempts per task. We then introduce two extensions: (1) database-level selection through population-based training to identify high-performing example collections, and (2) exemplar-level selection that retains individual trajectories based on their empirical utility as in-context examples. These extensions further enhance performance, achieving 91% on ALFWorld--matching more complex approaches that employ task-specific components and prompts. Our results demonstrate that automatic trajectory database construction offers a compelling alternative to labor-intensive knowledge engineering. 

---
# Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems 

**Authors**: Shaokun Zhang, Ming Yin, Jieyu Zhang, Jiale Liu, Zhiguang Han, Jingyang Zhang, Beibin Li, Chi Wang, Huazheng Wang, Yiran Chen, Qingyun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00212)  

**Abstract**: Failure attribution in LLM multi-agent systems-identifying the agent and step responsible for task failures-provides crucial clues for systems debugging but remains underexplored and labor-intensive. In this paper, we propose and formulate a new research area: automated failure attribution for LLM multi-agent systems. To support this initiative, we introduce the Who&When dataset, comprising extensive failure logs from 127 LLM multi-agent systems with fine-grained annotations linking failures to specific agents and decisive error steps. Using the Who&When, we develop and evaluate three automated failure attribution methods, summarizing their corresponding pros and cons. The best method achieves 53.5% accuracy in identifying failure-responsible agents but only 14.2% in pinpointing failure steps, with some methods performing below random. Even SOTA reasoning models, such as OpenAI o1 and DeepSeek R1, fail to achieve practical usability. These results highlight the task's complexity and the need for further research in this area. Code and dataset are available at this https URL 

---
# Detecting and Mitigating Hateful Content in Multimodal Memes with Vision-Language Models 

**Authors**: Minh-Hao Van, Xintao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00150)  

**Abstract**: The rapid evolution of social media has provided enhanced communication channels for individuals to create online content, enabling them to express their thoughts and opinions. Multimodal memes, often utilized for playful or humorous expressions with visual and textual elements, are sometimes misused to disseminate hate speech against individuals or groups. While the detection of hateful memes is well-researched, developing effective methods to transform hateful content in memes remains a significant challenge. Leveraging the powerful generation and reasoning capabilities of Vision-Language Models (VLMs), we address the tasks of detecting and mitigating hateful content. This paper presents two key contributions: first, a definition-guided prompting technique for detecting hateful memes, and second, a unified framework for mitigating hateful content in memes, named UnHateMeme, which works by replacing hateful textual and/or visual components. With our definition-guided prompts, VLMs achieve impressive performance on hateful memes detection task. Furthermore, our UnHateMeme framework, integrated with VLMs, demonstrates a strong capability to convert hateful memes into non-hateful forms that meet human-level criteria for hate speech and maintain multimodal coherence between image and text. Through empirical experiments, we show the effectiveness of state-of-the-art pretrained VLMs such as LLaVA, Gemini and GPT-4o on the proposed tasks, providing a comprehensive analysis of their respective strengths and limitations for these tasks. This paper aims to shed light on important applications of VLMs for ensuring safe and respectful online environments. 

---
# Optimization of embeddings storage for RAG systems using quantization and dimensionality reduction techniques 

**Authors**: Naamán Huerga-Pérez, Rubén Álvarez, Rubén Ferrero-Guillén, Alberto Martínez-Gutiérrez, Javier Díez-González  

**Link**: [PDF](https://arxiv.org/pdf/2505.00105)  

**Abstract**: Retrieval-Augmented Generation enhances language models by retrieving relevant information from external knowledge bases, relying on high-dimensional vector embeddings typically stored in float32 precision. However, storing these embeddings at scale presents significant memory challenges. To address this issue, we systematically investigate on MTEB benchmark two complementary optimization strategies: quantization, evaluating standard formats (float16, int8, binary) and low-bit floating-point types (float8), and dimensionality reduction, assessing methods like PCA, Kernel PCA, UMAP, Random Projections and Autoencoders. Our results show that float8 quantization achieves a 4x storage reduction with minimal performance degradation (<0.3%), significantly outperforming int8 quantization at the same compression level, being simpler to implement. PCA emerges as the most effective dimensionality reduction technique. Crucially, combining moderate PCA (e.g., retaining 50% dimensions) with float8 quantization offers an excellent trade-off, achieving 8x total compression with less performance impact than using int8 alone (which provides only 4x compression). To facilitate practical application, we propose a methodology based on visualizing the performance-storage trade-off space to identify the optimal configuration that maximizes performance within their specific memory constraints. 

---
# Humanizing LLMs: A Survey of Psychological Measurements with Tools, Datasets, and Human-Agent Applications 

**Authors**: Wenhan Dong, Yuemeng Zhao, Zhen Sun, Yule Liu, Zifan Peng, Jingyi Zheng, Zongmin Zhang, Ziyi Zhang, Jun Wu, Ruiming Wang, Shengmin Xu, Xinyi Huang, Xinlei He  

**Link**: [PDF](https://arxiv.org/pdf/2505.00049)  

**Abstract**: As large language models (LLMs) are increasingly used in human-centered tasks, assessing their psychological traits is crucial for understanding their social impact and ensuring trustworthy AI alignment. While existing reviews have covered some aspects of related research, several important areas have not been systematically discussed, including detailed discussions of diverse psychological tests, LLM-specific psychological datasets, and the applications of LLMs with psychological traits. To address this gap, we systematically review six key dimensions of applying psychological theories to LLMs: (1) assessment tools; (2) LLM-specific datasets; (3) evaluation metrics (consistency and stability); (4) empirical findings; (5) personality simulation methods; and (6) LLM-based behavior simulation. Our analysis highlights both the strengths and limitations of current methods. While some LLMs exhibit reproducible personality patterns under specific prompting schemes, significant variability remains across tasks and settings. Recognizing methodological challenges such as mismatches between psychological tools and LLMs' capabilities, as well as inconsistencies in evaluation practices, this study aims to propose future directions for developing more interpretable, robust, and generalizable psychological assessment frameworks for LLMs. 

---
