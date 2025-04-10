# OLMoTrace: Tracing Language Model Outputs Back to Trillions of Training Tokens 

**Authors**: Jiacheng Liu, Taylor Blanton, Yanai Elazar, Sewon Min, YenSung Chen, Arnavi Chheda-Kothary, Huy Tran, Byron Bischoff, Eric Marsh, Michael Schmitz, Cassidy Trier, Aaron Sarnat, Jenna James, Jon Borchardt, Bailey Kuehl, Evie Cheng, Karen Farley, Sruthi Sreeram, Taira Anderson, David Albright, Carissa Schoenick, Luca Soldaini, Dirk Groeneveld, Rock Yuren Pang, Pang Wei Koh, Noah A. Smith, Sophie Lebrecht, Yejin Choi, Hannaneh Hajishirzi, Ali Farhadi, Jesse Dodge  

**Link**: [PDF](https://arxiv.org/pdf/2504.07096)  

**Abstract**: We present OLMoTrace, the first system that traces the outputs of language models back to their full, multi-trillion-token training data in real time. OLMoTrace finds and shows verbatim matches between segments of language model output and documents in the training text corpora. Powered by an extended version of infini-gram (Liu et al., 2024), our system returns tracing results within a few seconds. OLMoTrace can help users understand the behavior of language models through the lens of their training data. We showcase how it can be used to explore fact checking, hallucination, and the creativity of language models. OLMoTrace is publicly available and fully open-source. 

---
# KG-LLM-Bench: A Scalable Benchmark for Evaluating LLM Reasoning on Textualized Knowledge Graphs 

**Authors**: Elan Markowitz, Krupa Galiya, Greg Ver Steeg, Aram Galstyan  

**Link**: [PDF](https://arxiv.org/pdf/2504.07087)  

**Abstract**: Knowledge graphs have emerged as a popular method for injecting up-to-date, factual knowledge into large language models (LLMs). This is typically achieved by converting the knowledge graph into text that the LLM can process in context. While multiple methods of encoding knowledge graphs have been proposed, the impact of this textualization process on LLM performance remains under-explored. We introduce KG-LLM-Bench, a comprehensive and extensible benchmark spanning five knowledge graph understanding tasks, and evaluate how different encoding strategies affect performance across various base models. Our extensive experiments with seven language models and five textualization strategies provide insights for optimizing LLM performance on KG reasoning tasks. 

---
# Self-Steering Language Models 

**Authors**: Gabriel Grand, Joshua B. Tenenbaum, Vikash K. Mansinghka, Alexander K. Lew, Jacob Andreas  

**Link**: [PDF](https://arxiv.org/pdf/2504.07081)  

**Abstract**: While test-time reasoning enables language models to tackle complex tasks, searching or planning in natural language can be slow, costly, and error-prone. But even when LMs struggle to emulate the precise reasoning steps needed to solve a problem, they often excel at describing its abstract structure--both how to verify solutions and how to search for them. This paper introduces DisCIPL, a method for "self-steering" LMs where a Planner model generates a task-specific inference program that is executed by a population of Follower models. Our approach equips LMs with the ability to write recursive search procedures that guide LM inference, enabling new forms of verifiable and efficient reasoning. When instantiated with a small Follower (e.g., Llama-3.2-1B), DisCIPL matches (and sometimes outperforms) much larger models, including GPT-4o and o1, on challenging constrained generation tasks. In decoupling planning from execution, our work opens up a design space of highly-parallelized Monte Carlo inference strategies that outperform standard best-of-N sampling, require no finetuning, and can be implemented automatically by existing LMs. 

---
# DeduCE: Deductive Consistency as a Framework to Evaluate LLM Reasoning 

**Authors**: Atharva Pandey, Kshitij Dubey, Rahul Sharma, Amit Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2504.07080)  

**Abstract**: Despite great performance on Olympiad-level reasoning problems, frontier large language models can still struggle on high school math when presented with novel problems outside standard benchmarks. Going beyond final accuracy, we propose a deductive consistency metric to analyze chain-of-thought output from language models (LMs).Formally, deductive reasoning involves two subtasks: understanding a set of input premises and inferring the conclusions that follow from them. The proposed metric studies LMs' performance on these subtasks, with the goal of explaining LMs' reasoning errors on novel problems: how well do LMs understand input premises with increasing context lengths, and how well can they infer conclusions over multiple reasoning hops? Since existing benchmarks may be memorized, we develop a pipeline to evaluate LMs' deductive consistency on novel, perturbed versions of benchmark problems. On novel grade school math problems (GSM-8k), we find that LMs are fairly robust to increasing number of input premises, but suffer significant accuracy decay as the number of reasoning hops is increased. Interestingly, these errors are masked in the original benchmark as all models achieve near 100% accuracy. As we increase the number of solution steps using a synthetic dataset, prediction over multiple hops still remains the major source of error compared to understanding input premises. Other factors, such as shifts in language style or natural propagation of early errors do not explain the trends. Our analysis provides a new view to characterize LM reasoning -- as computations over a window of input premises and reasoning hops -- that can provide unified evaluation across problem domains. 

---
# Kaleidoscope: In-language Exams for Massively Multilingual Vision Evaluation 

**Authors**: Israfel Salazar, Manuel Fernández Burda, Shayekh Bin Islam, Arshia Soltani Moakhar, Shivalika Singh, Fabian Farestam, Angelika Romanou, Danylo Boiko, Dipika Khullar, Mike Zhang, Dominik Krzemiński, Jekaterina Novikova, Luísa Shimabucoro, Joseph Marvin Imperial, Rishabh Maheshwary, Sharad Duwal, Alfonso Amayuelas, Swati Rajwal, Jebish Purbey, Ahmed Ruby, Nicholas Popovič, Marek Suppa, Azmine Toushik Wasi, Ram Mohan Rao Kadiyala, Olga Tsymboi, Maksim Kostritsya, Bardia Soltani Moakhar, Gabriel da Costa Merlin, Otávio Ferracioli Coletti, Maral Jabbari Shiviari, MohammadAmin farahani fard, Silvia Fernandez, María Grandury, Dmitry Abulkhanov, Drishti Sharma, Andre Guarnier De Mitri, Leticia Bossatto Marchezi, Johan Obando-Ceron, Nazar Kohut, Beyza Ermis, Desmond Elliott, Enzo Ferrante, Sara Hooker, Marzieh Fadaee  

**Link**: [PDF](https://arxiv.org/pdf/2504.07072)  

**Abstract**: The evaluation of vision-language models (VLMs) has mainly relied on English-language benchmarks, leaving significant gaps in both multilingual and multicultural coverage. While multilingual benchmarks have expanded, both in size and languages, many rely on translations of English datasets, failing to capture cultural nuances. In this work, we propose Kaleidoscope, as the most comprehensive exam benchmark to date for the multilingual evaluation of vision-language models. Kaleidoscope is a large-scale, in-language multimodal benchmark designed to evaluate VLMs across diverse languages and visual inputs. Kaleidoscope covers 18 languages and 14 different subjects, amounting to a total of 20,911 multiple-choice questions. Built through an open science collaboration with a diverse group of researchers worldwide, Kaleidoscope ensures linguistic and cultural authenticity. We evaluate top-performing multilingual vision-language models and find that they perform poorly on low-resource languages and in complex multimodal scenarios. Our results highlight the need for progress on culturally inclusive multimodal evaluation frameworks. 

---
# A Survey on Personalized and Pluralistic Preference Alignment in Large Language Models 

**Authors**: Zhouhang Xie, Junda Wu, Yiran Shen, Yu Xia, Xintong Li, Aaron Chang, Ryan Rossi, Sachin Kumar, Bodhisattwa Prasad Majumder, Jingbo Shang, Prithviraj Ammanabrolu, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2504.07070)  

**Abstract**: Personalized preference alignment for large language models (LLMs), the process of tailoring LLMs to individual users' preferences, is an emerging research direction spanning the area of NLP and personalization. In this survey, we present an analysis of works on personalized alignment and modeling for LLMs. We introduce a taxonomy of preference alignment techniques, including training time, inference time, and additionally, user-modeling based methods. We provide analysis and discussion on the strengths and limitations of each group of techniques and then cover evaluation, benchmarks, as well as open problems in the field. 

---
# HalluciNot: Hallucination Detection Through Context and Common Knowledge Verification 

**Authors**: Bibek Paudel, Alexander Lyzhov, Preetam Joshi, Puneet Anand  

**Link**: [PDF](https://arxiv.org/pdf/2504.07069)  

**Abstract**: This paper introduces a comprehensive system for detecting hallucinations in large language model (LLM) outputs in enterprise settings. We present a novel taxonomy of LLM responses specific to hallucination in enterprise applications, categorizing them into context-based, common knowledge, enterprise-specific, and innocuous statements. Our hallucination detection model HDM-2 validates LLM responses with respect to both context and generally known facts (common knowledge). It provides both hallucination scores and word-level annotations, enabling precise identification of problematic content. To evaluate it on context-based and common-knowledge hallucinations, we introduce a new dataset HDMBench. Experimental results demonstrate that HDM-2 out-performs existing approaches across RagTruth, TruthfulQA, and HDMBench datasets. This work addresses the specific challenges of enterprise deployment, including computational efficiency, domain specialization, and fine-grained error identification. Our evaluation dataset, model weights, and inference code are publicly available. 

---
# TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling 

**Authors**: Liang-Hsuan Tseng, Yi-Chang Chen, Kuan-Yi Lee, Da-Shan Shiu, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.07053)  

**Abstract**: Large Language Models (LLMs) excel in text-based natural language processing tasks but remain constrained by their reliance on textual inputs and outputs. To enable more natural human-LLM interaction, recent progress have focused on deriving a spoken language model (SLM) that can not only listen but also generate speech. To achieve this, a promising direction is to conduct speech-text joint modeling. However, recent SLM still lag behind text LLM due to the modality mismatch. One significant mismatch can be the sequence lengths between speech and text tokens. To address this, we introduce Text-Aligned Speech Tokenization and Embedding (TASTE), a method that directly addresses the modality gap by aligning speech token with the corresponding text transcription during the tokenization stage. We propose a method that can achieve this through the special aggregation mechanism and with speech reconstruction as the training objective. We conduct extensive experiments and show that TASTE can preserve essential paralinguistic information while dramatically reducing the token sequence length. Furthermore, by leveraging TASTE, we can adapt text-based LLMs into effective SLMs with parameter-efficient fine-tuning techniques such as Low-Rank Adaptation (LoRA). Experimental results on benchmark tasks, including SALMON and StoryCloze, demonstrate that TASTE-based SLMs perform similarly to previous full-finetuning methods. To our knowledge, TASTE is the first end-to-end approach that utilizes a reconstruction objective to automatically learn a text-aligned speech tokenization and embedding suitable for spoken language modeling. Our demo, code, and models are publicly available at this https URL. 

---
# Data Augmentation and Hyperparameter Tuning for Low-Resource MFA 

**Authors**: Alessio Tosolini, Claire Bowern  

**Link**: [PDF](https://arxiv.org/pdf/2504.07024)  

**Abstract**: A continued issue for those working with computational tools and endangered and under-resourced languages is the lower accuracy of results for languages with smaller amounts of data. We attempt to ameliorate this issue by using data augmentation methods to increase corpus size, comparing augmentation to hyperparameter tuning for multilingual forced alignment. Unlike text augmentation methods, audio augmentation does not lead to substantially increased performance. Hyperparameter tuning, on the other hand, results in substantial improvement without (for this amount of data) infeasible additional training time. For languages with small to medium amounts of training data, this is a workable alternative to adapting models from high-resource languages. 

---
# Evaluating Retrieval Augmented Generative Models for Document Queries in Transportation Safety 

**Authors**: Chad Melton, Alex Sorokine, Steve Peterson  

**Link**: [PDF](https://arxiv.org/pdf/2504.07022)  

**Abstract**: Applications of generative Large Language Models LLMs are rapidly expanding across various domains, promising significant improvements in workflow efficiency and information retrieval. However, their implementation in specialized, high-stakes domains such as hazardous materials transportation is challenging due to accuracy and reliability concerns. This study evaluates the performance of three fine-tuned generative models, ChatGPT, Google's Vertex AI, and ORNL Retrieval Augmented Generation augmented LLaMA 2 and LLaMA in retrieving regulatory information essential for hazardous material transportation compliance in the United States. Utilizing approximately 40 publicly available federal and state regulatory documents, we developed 100 realistic queries relevant to route planning and permitting requirements. Responses were qualitatively rated based on accuracy, detail, and relevance, complemented by quantitative assessments of semantic similarity between model outputs. Results demonstrated that the RAG-augmented LLaMA models significantly outperformed Vertex AI and ChatGPT, providing more detailed and generally accurate information, despite occasional inconsistencies. This research introduces the first known application of RAG in transportation safety, emphasizing the need for domain-specific fine-tuning and rigorous evaluation methodologies to ensure reliability and minimize the risk of inaccuracies in high-stakes environments. 

---
# Towards LLMs Robustness to Changes in Prompt Format Styles 

**Authors**: Lilian Ngweta, Kiran Kate, Jason Tsay, Yara Rizk  

**Link**: [PDF](https://arxiv.org/pdf/2504.06969)  

**Abstract**: Large language models (LLMs) have gained popularity in recent years for their utility in various applications. However, they are sensitive to non-semantic changes in prompt formats, where small changes in the prompt format can lead to significant performance fluctuations. In the literature, this problem is commonly referred to as prompt brittleness. Previous research on prompt engineering has focused mainly on developing techniques for identifying the optimal prompt for specific tasks. Some studies have also explored the issue of prompt brittleness and proposed methods to quantify performance variations; however, no simple solution has been found to address this challenge. We propose Mixture of Formats (MOF), a simple and efficient technique for addressing prompt brittleness in LLMs by diversifying the styles used in the prompt few-shot examples. MOF was inspired by computer vision techniques that utilize diverse style datasets to prevent models from associating specific styles with the target variable. Empirical results show that our proposed technique reduces style-induced prompt brittleness in various LLMs while also enhancing overall performance across prompt variations and different datasets. 

---
# RuOpinionNE-2024: Extraction of Opinion Tuples from Russian News Texts 

**Authors**: Natalia Loukachevitch, Natalia Tkachenko, Anna Lapanitsyna, Mikhail Tikhomirov, Nicolay Rusnachenko  

**Link**: [PDF](https://arxiv.org/pdf/2504.06947)  

**Abstract**: In this paper, we introduce the Dialogue Evaluation shared task on extraction of structured opinions from Russian news texts. The task of the contest is to extract opinion tuples for a given sentence; the tuples are composed of a sentiment holder, its target, an expression and sentiment from the holder to the target. In total, the task received more than 100 submissions. The participants experimented mainly with large language models in zero-shot, few-shot and fine-tuning formats. The best result on the test set was obtained with fine-tuning of a large language model. We also compared 30 prompts and 11 open source language models with 3-32 billion parameters in the 1-shot and 10-shot settings and found the best models and prompts. 

---
# Data Augmentation for Fake Reviews Detection in Multiple Languages and Multiple Domains 

**Authors**: Ming Liu, Massimo Poesio  

**Link**: [PDF](https://arxiv.org/pdf/2504.06917)  

**Abstract**: With the growth of the Internet, buying habits have changed, and customers have become more dependent on the online opinions of other customers to guide their purchases. Identifying fake reviews thus became an important area for Natural Language Processing (NLP) research. However, developing high-performance NLP models depends on the availability of large amounts of training data, which are often not available for low-resource languages or domains. In this research, we used large language models to generate datasets to train fake review detectors. Our approach was used to generate fake reviews in different domains (book reviews, restaurant reviews, and hotel reviews) and different languages (English and Chinese). Our results demonstrate that our data augmentation techniques result in improved performance at fake review detection for all domains and languages. The accuracy of our fake review detection model can be improved by 0.3 percentage points on DeRev TEST, 10.9 percentage points on Amazon TEST, 8.3 percentage points on Yelp TEST and 7.2 percentage points on DianPing TEST using the augmented datasets. 

---
# Identifying Aspects in Peer Reviews 

**Authors**: Sheng Lu, Ilia Kuznetsov, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2504.06910)  

**Abstract**: Peer review is central to academic publishing, but the growing volume of submissions is straining the process. This motivates the development of computational approaches to support peer review. While each review is tailored to a specific paper, reviewers often make assessments according to certain aspects such as Novelty, which reflect the values of the research community. This alignment creates opportunities for standardizing the reviewing process, improving quality control, and enabling computational support. While prior work has demonstrated the potential of aspect analysis for peer review assistance, the notion of aspect remains poorly formalized. Existing approaches often derive aspect sets from review forms and guidelines of major NLP venues, yet data-driven methods for aspect identification are largely underexplored. To address this gap, our work takes a bottom-up approach: we propose an operational definition of aspect and develop a data-driven schema for deriving fine-grained aspects from a corpus of peer reviews. We introduce a dataset of peer reviews augmented with aspects and show how it can be used for community-level review analysis. We further show how the choice of aspects can impact downstream applications, such as LLM-generated review detection. Our results lay a foundation for a principled and data-driven investigation of review aspects, and pave the path for new applications of NLP to support peer review. 

---
# Persona Dynamics: Unveiling the Impact of Personality Traits on Agents in Text-Based Games 

**Authors**: Seungwon Lim, Seungbeen Lee, Dongjun Min, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06868)  

**Abstract**: Artificial agents are increasingly central to complex interactions and decision-making tasks, yet aligning their behaviors with desired human values remains an open challenge. In this work, we investigate how human-like personality traits influence agent behavior and performance within text-based interactive environments. We introduce PANDA: PersonalityAdapted Neural Decision Agents, a novel method for projecting human personality traits onto agents to guide their behavior. To induce personality in a text-based game agent, (i) we train a personality classifier to identify what personality type the agent's actions exhibit, and (ii) we integrate the personality profiles directly into the agent's policy-learning pipeline. By deploying agents embodying 16 distinct personality types across 25 text-based games and analyzing their trajectories, we demonstrate that an agent's action decisions can be guided toward specific personality profiles. Moreover, certain personality types, such as those characterized by higher levels of Openness, display marked advantages in performance. These findings underscore the promise of personality-adapted agents for fostering more aligned, effective, and human-centric decision-making in interactive environments. 

---
# Integrating Cognitive Processing Signals into Language Models: A Review of Advances, Applications and Future Directions 

**Authors**: Angela Lopez-Cardona, Sebastian Idesis, Ioannis Arapakis  

**Link**: [PDF](https://arxiv.org/pdf/2504.06843)  

**Abstract**: Recently, the integration of cognitive neuroscience in Natural Language Processing (NLP) has gained significant attention. This article provides a critical and timely overview of recent advancements in leveraging cognitive signals, particularly Eye-tracking (ET) signals, to enhance Language Models (LMs) and Multimodal Large Language Models (MLLMs). By incorporating user-centric cognitive signals, these approaches address key challenges, including data scarcity and the environmental costs of training large-scale models. Cognitive signals enable efficient data augmentation, faster convergence, and improved human alignment. The review emphasises the potential of ET data in tasks like Visual Question Answering (VQA) and mitigating hallucinations in MLLMs, and concludes by discussing emerging challenges and research trends. 

---
# Open Problems and a Hypothetical Path Forward in LLM Knowledge Paradigms 

**Authors**: Xiaotian Ye, Mengqi Zhang, Shu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06823)  

**Abstract**: Knowledge is fundamental to the overall capabilities of Large Language Models (LLMs). The knowledge paradigm of a model, which dictates how it encodes and utilizes knowledge, significantly affects its performance. Despite the continuous development of LLMs under existing knowledge paradigms, issues within these frameworks continue to constrain model potential.
This blog post highlight three critical open problems limiting model capabilities: (1) challenges in knowledge updating for LLMs, (2) the failure of reverse knowledge generalization (the reversal curse), and (3) conflicts in internal knowledge. We review recent progress made in addressing these issues and discuss potential general solutions. Based on observations in these areas, we propose a hypothetical paradigm based on Contextual Knowledge Scaling, and further outline implementation pathways that remain feasible within contemporary techniques. Evidence suggests this approach holds potential to address current shortcomings, serving as our vision for future model paradigms.
This blog post aims to provide researchers with a brief overview of progress in LLM knowledge systems, while provide inspiration for the development of next-generation model architectures. 

---
# Inducing Programmatic Skills for Agentic Tasks 

**Authors**: Zora Zhiruo Wang, Apurva Gandhi, Graham Neubig, Daniel Fried  

**Link**: [PDF](https://arxiv.org/pdf/2504.06821)  

**Abstract**: To succeed in common digital tasks such as web navigation, agents must carry out a variety of specialized tasks such as searching for products or planning a travel route. To tackle these tasks, agents can bootstrap themselves by learning task-specific skills online through interaction with the web environment. In this work, we demonstrate that programs are an effective representation for skills. We propose agent skill induction (ASI), which allows agents to adapt themselves by inducing, verifying, and utilizing program-based skills on the fly. We start with an evaluation on the WebArena agent benchmark and show that ASI outperforms the static baseline agent and its text-skill counterpart by 23.5% and 11.3% in success rate, mainly thanks to the programmatic verification guarantee during the induction phase. ASI also improves efficiency by reducing 10.7-15.3% of the steps over baselines, by composing primitive actions (e.g., click) into higher-level skills (e.g., search product). We then highlight the efficacy of ASI in remaining efficient and accurate under scaled-up web activities. Finally, we examine the generalizability of induced skills when transferring between websites, and find that ASI can effectively reuse common skills, while also updating incompatible skills to versatile website changes. 

---
# A Graph Diffusion Algorithm for Lexical Similarity Evaluation 

**Authors**: Karol Mikula, Mariana Sarkociová Remešíková  

**Link**: [PDF](https://arxiv.org/pdf/2504.06816)  

**Abstract**: In this paper, we present an algorithm for evaluating lexical similarity between a given language and several reference language clusters. As an input, we have a list of concepts and the corresponding translations in all considered languages. Moreover, each reference language is assigned to one of $c$ language clusters. For each of the concepts, the algorithm computes the distance between each pair of translations. Based on these distances, it constructs a weighted directed graph, where every vertex represents a language. After, it solves a graph diffusion equation with a Dirichlet boundary condition, where the unknown is a map from the vertex set to $\mathbb{R}^c$. The resulting coordinates are values from the interval $[0,1]$ and they can be interpreted as probabilities of belonging to each of the clusters or as a lexical similarity distribution with respect to the reference clusters. The distances between translations are calculated using phonetic transcriptions and a modification of the Damerau-Levenshtein distance. The algorithm can be useful in analyzing relationships between languages spoken in multilingual territories with a lot of mutual influences. We demonstrate this by presenting a case study regarding various European languages. 

---
# Domain-Specific Pruning of Large Mixture-of-Experts Models with Few-shot Demonstrations 

**Authors**: Zican Dong, Han Peng, Peiyu Liu, Wayne Xin Zhao, Dong Wu, Feng Xiao, Zhifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06792)  

**Abstract**: Mixture-of-Experts (MoE) models achieve a favorable trade-off between performance and inference efficiency by activating only a subset of experts. However, the memory overhead of storing all experts remains a major limitation, especially in large-scale MoE models such as DeepSeek-R1 (671B). In this study, we investigate domain specialization and expert redundancy in large-scale MoE models and uncover a consistent behavior we term few-shot expert localization, with only a few demonstrations, the model consistently activates a sparse and stable subset of experts. Building on this observation, we propose a simple yet effective pruning framework, EASY-EP, that leverages a few domain-specific demonstrations to identify and retain only the most relevant experts. EASY-EP comprises two key components: output-aware expert importance assessment and expert-level token contribution estimation. The former evaluates the importance of each expert for the current token by considering the gating scores and magnitudes of the outputs of activated experts, while the latter assesses the contribution of tokens based on representation similarities after and before routed experts. Experiments show that our method can achieve comparable performances and $2.99\times$ throughput under the same memory budget with full DeepSeek-R1 with only half the experts. Our code is available at this https URL. 

---
# NLP Security and Ethics, in the Wild 

**Authors**: Heather Lent, Erick Galinkin, Yiyi Chen, Jens Myrup Pedersen, Leon Derczynski, Johannes Bjerva  

**Link**: [PDF](https://arxiv.org/pdf/2504.06669)  

**Abstract**: As NLP models are used by a growing number of end-users, an area of increasing importance is NLP Security (NLPSec): assessing the vulnerability of models to malicious attacks and developing comprehensive countermeasures against them. While work at the intersection of NLP and cybersecurity has the potential to create safer NLP for all, accidental oversights can result in tangible harm (e.g., breaches of privacy or proliferation of malicious models). In this emerging field, however, the research ethics of NLP have not yet faced many of the long-standing conundrums pertinent to cybersecurity, until now. We thus examine contemporary works across NLPSec, and explore their engagement with cybersecurity's ethical norms. We identify trends across the literature, ultimately finding alarming gaps on topics like harm minimization and responsible disclosure. To alleviate these concerns, we provide concrete recommendations to help NLP researchers navigate this space more ethically, bridging the gap between traditional cybersecurity and NLP ethics, which we frame as ``white hat NLP''. The goal of this work is to help cultivate an intentional culture of ethical research for those working in NLP Security. 

---
# SEE: Continual Fine-tuning with Sequential Ensemble of Experts 

**Authors**: Zhilin Wang, Yafu Li, Xiaoye Qu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.06664)  

**Abstract**: Continual fine-tuning of large language models (LLMs) suffers from catastrophic forgetting. Rehearsal-based methods mitigate this problem by retaining a small set of old data. Nevertheless, they still suffer inevitable performance loss. Although training separate experts for each task can help prevent forgetting, effectively assembling them remains a challenge. Some approaches use routers to assign tasks to experts, but in continual learning, they often require retraining for optimal performance. To address these challenges, we introduce the Sequential Ensemble of Experts (SEE) framework. SEE removes the need for an additional router, allowing each expert to independently decide whether a query should be handled. The framework employs distributed routing, and during continual fine-tuning, SEE only requires the training of new experts for incoming tasks rather than retraining the entire system. Experiments reveal that the SEE outperforms prior approaches, including multi-task learning, in continual fine-tuning. It also demonstrates remarkable generalization ability, as the expert can effectively identify out-of-distribution queries, which can then be directed to a more generalized model for resolution. This work highlights the promising potential of integrating routing and response mechanisms within each expert, paving the way for the future of distributed model ensembling. 

---
# ThoughtProbe: Classifier-Guided Thought Space Exploration Leveraging LLM Intrinsic Reasoning 

**Authors**: Zijian Wang, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06650)  

**Abstract**: Pre-trained large language models (LLMs) have been demonstrated to possess intrinsic reasoning capabilities that can emerge naturally when expanding the response space. However, the neural representation mechanisms underlying these intrinsic capabilities and approaches for their optimal utilization remain inadequately understood. In this work, we make the key discovery that a simple linear classifier can effectively detect intrinsic reasoning capabilities in LLMs' activation space, particularly within specific representation types and network layers. Based on this finding, we propose a classifier-guided search framework that strategically explore a tree-structured response space. In each node expansion, the classifier serves as a scoring and ranking mechanism that efficiently allocates computational resources by identifying and prioritizing more thoughtful reasoning directions for continuation. After completing the tree expansion, we collect answers from all branches to form a candidate answer pool. We propose a branch-aggregation selection method that marginalizes over all supporting branches by aggregating their thoughtfulness scores, thereby identifying the optimal answer from the pool. Experimental results show that our framework's comprehensive exploration not only covers valid reasoning chains but also effectively identifies them, achieving significant improvements across multiple arithmetic reasoning benchmarks. 

---
# Automated Business Process Analysis: An LLM-Based Approach to Value Assessment 

**Authors**: William De Michele, Abel Armas Cervantes, Lea Frermann  

**Link**: [PDF](https://arxiv.org/pdf/2504.06600)  

**Abstract**: Business processes are fundamental to organizational operations, yet their optimization remains challenging due to the timeconsuming nature of manual process analysis. Our paper harnesses Large Language Models (LLMs) to automate value-added analysis, a qualitative process analysis technique that aims to identify steps in the process that do not deliver value. To date, this technique is predominantly manual, time-consuming, and subjective. Our method offers a more principled approach which operates in two phases: first, decomposing high-level activities into detailed steps to enable granular analysis, and second, performing a value-added analysis to classify each step according to Lean principles. This approach enables systematic identification of waste while maintaining the semantic understanding necessary for qualitative analysis. We develop our approach using 50 business process models, for which we collect and publish manual ground-truth labels. Our evaluation, comparing zero-shot baselines with more structured prompts reveals (a) a consistent benefit of structured prompting and (b) promising performance for both tasks. We discuss the potential for LLMs to augment human expertise in qualitative process analysis while reducing the time and subjectivity inherent in manual approaches. 

---
# Bypassing Safety Guardrails in LLMs Using Humor 

**Authors**: Pedro Cisneros-Velarde  

**Link**: [PDF](https://arxiv.org/pdf/2504.06577)  

**Abstract**: In this paper, we show it is possible to bypass the safety guardrails of large language models (LLMs) through a humorous prompt including the unsafe request. In particular, our method does not edit the unsafe request and follows a fixed template -- it is simple to implement and does not need additional LLMs to craft prompts. Extensive experiments show the effectiveness of our method across different LLMs. We also show that both removing and adding more humor to our method can reduce its effectiveness -- excessive humor possibly distracts the LLM from fulfilling its unsafe request. Thus, we argue that LLM jailbreaking occurs when there is a proper balance between focus on the unsafe request and presence of humor. 

---
# Do Reasoning Models Show Better Verbalized Calibration? 

**Authors**: Qingcheng Zeng, Weihao Xuan, Leyang Cui, Rob Voigt  

**Link**: [PDF](https://arxiv.org/pdf/2504.06564)  

**Abstract**: Large reasoning models (LRMs) have recently shown impressive capabilities in complex reasoning by leveraging increased test-time computation and exhibiting behaviors akin to human-like deliberation. Despite these advances, it remains an open question whether LRMs are better calibrated - particularly in their verbalized confidence - compared to instruction-tuned counterparts. In this paper, we investigate the calibration properties of LRMs trained via supervised fine-tuning distillation on long reasoning traces (henceforth SFT reasoning models) and outcome-based reinforcement learning for reasoning (henceforth RL reasoning models) across diverse domains. Our findings reveal that LRMs significantly outperform instruction-tuned models on complex reasoning tasks in both accuracy and confidence calibration. In contrast, we find surprising trends in the domain of factuality in particular. On factuality tasks, while Deepseek-R1 shows strong calibration behavior, smaller QwQ-32B shows no improvement over instruct models; moreover, SFT reasoning models display worse calibration (greater overconfidence) compared to instruct models. Our results provide evidence for a potentially critical role of reasoning-oriented RL training in improving LLMs' capacity for generating trustworthy, self-aware outputs. 

---
# FuseRL: Dense Preference Optimization for Heterogeneous Model Fusion 

**Authors**: Longguang Zhong, Fanqi Wan, Ziyi Yang, Guosheng Liang, Tianyuan Shi, Xiaojun Quan  

**Link**: [PDF](https://arxiv.org/pdf/2504.06562)  

**Abstract**: Heterogeneous model fusion enhances the performance of LLMs by integrating the knowledge and capabilities of multiple structurally diverse models. However, existing approaches often rely solely on selecting the best output for each prompt from source models, which underutilizes their full potential due to limited source knowledge and results in sparse optimization signals. To address this limitation, we propose FuseRL, a novel two-stage framework comprising FuseSFT and FusePO to maximize the utilization of source LLMs. FuseSFT establishes a robust initialization by integrating the strengths of heterogeneous source models through weighted supervised fine-tuning (SFT) on diverse outputs for each prompt. FusePO optimizes weighted preferences based on the outputs of multiple source models to enable superior alignment performance. Extensive experiments demonstrate the effectiveness of our framework across various preference alignment methods, including RLOO, DPO, and SimPO. Using Llama-3.1-8B-Instruct as the target model, our approach achieves state-of-the-art performance among 8B LLMs on the AlpacaEval-2 and Arena-Hard benchmarks. Further analysis suggests that FuseSFT regularizes the training process to reduce overfitting, while FusePO introduces dense and diverse signals for preference optimization. 

---
# NeedleInATable: Exploring Long-Context Capability of Large Language Models towards Long-Structured Tables 

**Authors**: Lanrui Wang, Mingyu Zheng, Hongyin Tang, Zheng Lin, Yanan Cao, Jingang Wang, Xunliang Cai, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06560)  

**Abstract**: Processing structured tabular data, particularly lengthy tables, constitutes a fundamental yet challenging task for large language models (LLMs). However, existing long-context benchmarks primarily focus on unstructured text, neglecting the challenges of long and complex structured tables. To address this gap, we introduce NeedleInATable (NIAT), a novel task that treats each table cell as a "needle" and requires the model to extract the target cell under different queries. Evaluation results of mainstream LLMs on this benchmark show they lack robust long-table comprehension, often relying on superficial correlations or shortcuts for complex table understanding tasks, revealing significant limitations in processing intricate tabular data. To this end, we propose a data synthesis method to enhance models' long-table comprehension capabilities. Experimental results show that our synthesized training data significantly enhances LLMs' performance on the NIAT task, outperforming both long-context LLMs and long-table agent methods. This work advances the evaluation of LLMs' genuine long-structured table comprehension capabilities and paves the way for progress in long-context and table understanding applications. 

---
# Lugha-Llama: Adapting Large Language Models for African Languages 

**Authors**: Happy Buzaaba, Alexander Wettig, David Ifeoluwa Adelani, Christiane Fellbaum  

**Link**: [PDF](https://arxiv.org/pdf/2504.06536)  

**Abstract**: Large language models (LLMs) have achieved impressive results in a wide range of natural language applications. However, they often struggle to recognize low-resource languages, in particular African languages, which are not well represented in large training corpora. In this paper, we consider how to adapt LLMs to low-resource African languages. We find that combining curated data from African languages with high-quality English educational texts results in a training mix that substantially improves the model's performance on these languages. On the challenging IrokoBench dataset, our models consistently achieve the best performance amongst similarly sized baselines, particularly on knowledge-intensive multiple-choice questions (AfriMMLU). Additionally, on the cross-lingual question answering benchmark AfriQA, our models outperform the base model by over 10%. To better understand the role of English data during training, we translate a subset of 200M tokens into Swahili language and perform an analysis which reveals that the content of these data is primarily responsible for the strong performance. We release our models and data to encourage future research on African languages. 

---
# CDER: Collaborative Evidence Retrieval for Document-level Relation Extraction 

**Authors**: Khai Phan Tran, Xue Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.06529)  

**Abstract**: Document-level Relation Extraction (DocRE) involves identifying relations between entities across multiple sentences in a document. Evidence sentences, crucial for precise entity pair relationships identification, enhance focus on essential text segments, improving DocRE performance. However, existing evidence retrieval systems often overlook the collaborative nature among semantically similar entity pairs in the same document, hindering the effectiveness of the evidence retrieval task. To address this, we propose a novel evidence retrieval framework, namely CDER. CDER employs an attentional graph-based architecture to capture collaborative patterns and incorporates a dynamic sub-structure for additional robustness in evidence retrieval. Experimental results on the benchmark DocRE dataset show that CDER not only excels in the evidence retrieval task but also enhances overall performance of existing DocRE system. 

---
# Analyzing Examinee Comments using DistilBERT and Machine Learning to Ensure Quality Control in Exam Content 

**Authors**:   

**Link**: [PDF](https://arxiv.org/pdf/2504.06465)  

**Abstract**: This study explores using Natural Language Processing (NLP) to analyze candidate comments for identifying problematic test items. We developed and validated machine learning models that automatically identify relevant negative feedback, evaluated approaches of incorporating psychometric features enhances model performance, and compared NLP-flagged items with traditionally flagged items. Results demonstrate that candidate feedback provides valuable complementary information to statistical methods, potentially improving test validity while reducing manual review burden. This research offers testing organizations an efficient mechanism to incorporate direct candidate experience into quality assurance processes. 

---
# Can LLMs Simulate Personas with Reversed Performance? A Benchmark for Counterfactual Instruction Following 

**Authors**: Sai Adith Senthil Kumar, Hao Yan, Saipavan Perepa, Murong Yue, Ziyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.06460)  

**Abstract**: Large Language Models (LLMs) are now increasingly widely used to simulate personas in virtual environments, leveraging their instruction-following capability. However, we discovered that even state-of-the-art LLMs cannot simulate personas with reversed performance (e.g., student personas with low proficiency in educational settings), which impairs the simulation diversity and limits the practical applications of the simulated environments. In this work, using mathematical reasoning as a representative scenario, we propose the first benchmark dataset for evaluating LLMs on simulating personas with reversed performance, a capability that we dub "counterfactual instruction following". We evaluate both open-weight and closed-source LLMs on this task and find that LLMs, including the OpenAI o1 reasoning model, all struggle to follow counterfactual instructions for simulating reversedly performing personas. Intersectionally simulating both the performance level and the race population of a persona worsens the effect even further. These results highlight the challenges of counterfactual instruction following and the need for further research. 

---
# Don't Let It Hallucinate: Premise Verification via Retrieval-Augmented Logical Reasoning 

**Authors**: Yuehan Qin, Shawn Li, Yi Nian, Xinyan Velocity Yu, Yue Zhao, Xuezhe Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.06438)  

**Abstract**: Large language models (LLMs) have shown substantial capacity for generating fluent, contextually appropriate responses. However, they can produce hallucinated outputs, especially when a user query includes one or more false premises-claims that contradict established facts. Such premises can mislead LLMs into offering fabricated or misleading details. Existing approaches include pretraining, fine-tuning, and inference-time techniques that often rely on access to logits or address hallucinations after they occur. These methods tend to be computationally expensive, require extensive training data, or lack proactive mechanisms to prevent hallucination before generation, limiting their efficiency in real-time applications. We propose a retrieval-based framework that identifies and addresses false premises before generation. Our method first transforms a user's query into a logical representation, then applies retrieval-augmented generation (RAG) to assess the validity of each premise using factual sources. Finally, we incorporate the verification results into the LLM's prompt to maintain factual consistency in the final output. Experiments show that this approach effectively reduces hallucinations, improves factual accuracy, and does not require access to model logits or large-scale fine-tuning. 

---
# Language-Dependent Political Bias in AI: A Study of ChatGPT and Gemini 

**Authors**: Dogus Yuksel, Mehmet Cem Catalbas, Bora Oc  

**Link**: [PDF](https://arxiv.org/pdf/2504.06436)  

**Abstract**: As leading examples of large language models, ChatGPT and Gemini claim to provide accurate and unbiased information, emphasizing their commitment to political neutrality and avoidance of personal bias. This research investigates the political tendency of large language models and the existence of differentiation according to the query language. For this purpose, ChatGPT and Gemini were subjected to a political axis test using 14 different languages. The findings of the study suggest that these large language models do exhibit political tendencies, with both models demonstrating liberal and leftist biases. A comparative analysis revealed that Gemini exhibited a more pronounced liberal and left-wing tendency compared to ChatGPT. The study also found that these political biases varied depending on the language used for inquiry. The study delves into the factors that constitute political tendencies and linguistic differentiation, exploring differences in the sources and scope of educational data, structural and grammatical features of languages, cultural and political contexts, and the model's response to linguistic features. From this standpoint, and an ethical perspective, it is proposed that artificial intelligence tools should refrain from asserting a lack of political tendencies and neutrality, instead striving for political neutrality and executing user queries by incorporating these tendencies. 

---
# S'MoRE: Structural Mixture of Residual Experts for LLM Fine-tuning 

**Authors**: Hanqing Zeng, Yinglong Xia, Zhuokai Zhao, Gilbert Jiang, Qiang Zhang, Jiayi Liu, Lizhu Zhang, Xiangjun Fan, Benyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06426)  

**Abstract**: Fine-tuning pre-trained large language models (LLMs) presents a dual challenge of balancing parameter efficiency and model capacity. Existing methods like low-rank adaptations (LoRA) are efficient but lack flexibility, while Mixture-of-Experts (MoE) architectures enhance model capacity at the cost of more & under-utilized parameters. To address these limitations, we propose Structural Mixture of Residual Experts (S'MoRE), a novel framework that seamlessly integrates the efficiency of LoRA with the flexibility of MoE. Specifically, S'MoRE employs hierarchical low-rank decomposition of expert weights, yielding residuals of varying orders interconnected in a multi-layer structure. By routing input tokens through sub-trees of residuals, S'MoRE emulates the capacity of many experts by instantiating and assembling just a few low-rank matrices. We craft the inter-layer propagation of S'MoRE's residuals as a special type of Graph Neural Network (GNN), and prove that under similar parameter budget, S'MoRE improves "structural flexibility" of traditional MoE (or Mixture-of-LoRA) by exponential order. Comprehensive theoretical analysis and empirical results demonstrate that S'MoRE achieves superior fine-tuning performance, offering a transformative approach for efficient LLM adaptation. 

---
# The Zero Body Problem: Probing LLM Use of Sensory Language 

**Authors**: Rebecca M. M. Hicke, Sil Hamilton, David Mimno  

**Link**: [PDF](https://arxiv.org/pdf/2504.06393)  

**Abstract**: Sensory language expresses embodied experiences ranging from taste and sound to excitement and stomachache. This language is of interest to scholars from a wide range of domains including robotics, narratology, linguistics, and cognitive science. In this work, we explore whether language models, which are not embodied, can approximate human use of embodied language. We extend an existing corpus of parallel human and model responses to short story prompts with an additional 18,000 stories generated by 18 popular models. We find that all models generate stories that differ significantly from human usage of sensory language, but the direction of these differences varies considerably between model families. Namely, Gemini models use significantly more sensory language than humans along most axes whereas most models from the remaining five families use significantly less. Linear probes run on five models suggest that they are capable of identifying sensory language. However, we find preliminary evidence suggesting that instruction tuning may discourage usage of sensory language. Finally, to support further work, we release our expanded story dataset. 

---
# Query Understanding in LLM-based Conversational Information Seeking 

**Authors**: Yifei Yuan, Zahra Abbasiantaeb, Yang Deng, Mohammad Aliannejadi  

**Link**: [PDF](https://arxiv.org/pdf/2504.06356)  

**Abstract**: Query understanding in Conversational Information Seeking (CIS) involves accurately interpreting user intent through context-aware interactions. This includes resolving ambiguities, refining queries, and adapting to evolving information needs. Large Language Models (LLMs) enhance this process by interpreting nuanced language and adapting dynamically, improving the relevance and precision of search results in real-time. In this tutorial, we explore advanced techniques to enhance query understanding in LLM-based CIS systems. We delve into LLM-driven methods for developing robust evaluation metrics to assess query understanding quality in multi-turn interactions, strategies for building more interactive systems, and applications like proactive query management and query reformulation. We also discuss key challenges in integrating LLMs for query understanding in conversational search systems and outline future research directions. Our goal is to deepen the audience's understanding of LLM-based conversational query understanding and inspire discussions to drive ongoing advancements in this field. 

---
# Reducing Formal Context Extraction: A Newly Proposed Framework from Big Corpora 

**Authors**: Bryar A. Hassan, Shko M. Qader, Alla A. Hassan, Joan Lu, Aram M. Ahmed, Jafar Majidpour, Tarik A. Rashid  

**Link**: [PDF](https://arxiv.org/pdf/2504.06285)  

**Abstract**: Automating the extraction of concept hierarchies from free text is advantageous because manual generation is frequently labor- and resource-intensive. Free result, the whole procedure for concept hierarchy learning from free text entails several phases, including sentence-level text processing, sentence splitting, and tokenization. Lemmatization is after formal context analysis (FCA) to derive the pairings. Nevertheless, there could be a few uninteresting and incorrect pairings in the formal context. It may take a while to generate formal context; thus, size reduction formal context is necessary to weed out irrelevant and incorrect pairings to extract the concept lattice and hierarchies more quickly. This study aims to propose a framework for reducing formal context in extracting concept hierarchies from free text to reduce the ambiguity of the formal context. We achieve this by reducing the size of the formal context using a hybrid of a WordNet-based method and a frequency-based technique. Using 385 samples from the Wikipedia corpus and the suggested framework, tests are carried out to examine the reduced size of formal context, leading to concept lattice and concept hierarchy. With the help of concept lattice-invariants, the generated formal context lattice is compared to the normal one. In contrast to basic ones, the homomorphic between the resultant lattices retains up to 98% of the quality of the generating concept hierarchies, and the reduced concept lattice receives the structural connection of the standard one. Additionally, the new framework is compared to five baseline techniques to calculate the running time on random datasets with various densities. The findings demonstrate that, in various fill ratios, hybrid approaches of the proposed method outperform other indicated competing strategies in concept lattice performance. 

---
# Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning 

**Authors**: Nikhil Shivakumar Nayak, Krishnateja Killamsetty, Ligong Han, Abhishek Bhandwaldar, Prateek Chanda, Kai Xu, Hao Wang, Aldo Pareja, Oleg Silkin, Mustafa Eyceoz, Akash Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2504.07097)  

**Abstract**: Continual learning in large language models (LLMs) is prone to catastrophic forgetting, where adapting to new tasks significantly degrades performance on previously learned ones. Existing methods typically rely on low-rank, parameter-efficient updates that limit the model's expressivity and introduce additional parameters per task, leading to scalability issues. To address these limitations, we propose a novel continual full fine-tuning approach leveraging adaptive singular value decomposition (SVD). Our method dynamically identifies task-specific low-rank parameter subspaces and constrains updates to be orthogonal to critical directions associated with prior tasks, thus effectively minimizing interference without additional parameter overhead or storing previous task gradients. We evaluate our approach extensively on standard continual learning benchmarks using both encoder-decoder (T5-Large) and decoder-only (LLaMA-2 7B) models, spanning diverse tasks including classification, generation, and reasoning. Empirically, our method achieves state-of-the-art results, up to 7% higher average accuracy than recent baselines like O-LoRA, and notably maintains the model's general linguistic capabilities, instruction-following accuracy, and safety throughout the continual learning process by reducing forgetting to near-negligible levels. Our adaptive SVD framework effectively balances model plasticity and knowledge retention, providing a practical, theoretically grounded, and computationally scalable solution for continual learning scenarios in large language models. 

---
# OmniCaptioner: One Captioner to Rule Them All 

**Authors**: Yiting Lu, Jiakang Yuan, Zhen Li, Shitian Zhao, Qi Qin, Xinyue Li, Le Zhuo, Licheng Wen, Dongyang Liu, Yuewen Cao, Xiangchao Yan, Xin Li, Botian Shi, Tao Chen, Zhibo Chen, Lei Bai, Bo Zhang, Peng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.07089)  

**Abstract**: We propose OmniCaptioner, a versatile visual captioning framework for generating fine-grained textual descriptions across a wide variety of visual domains. Unlike prior methods limited to specific image types (e.g., natural images or geometric visuals), our framework provides a unified solution for captioning natural images, visual text (e.g., posters, UIs, textbooks), and structured visuals (e.g., documents, tables, charts). By converting low-level pixel information into semantically rich textual representations, our framework bridges the gap between visual and textual modalities. Our results highlight three key advantages: (i) Enhanced Visual Reasoning with LLMs, where long-context captions of visual modalities empower LLMs, particularly the DeepSeek-R1 series, to reason effectively in multimodal scenarios; (ii) Improved Image Generation, where detailed captions improve tasks like text-to-image generation and image transformation; and (iii) Efficient Supervised Fine-Tuning (SFT), which enables faster convergence with less data. We believe the versatility and adaptability of OmniCaptioner can offer a new perspective for bridging the gap between language and visual modalities. 

---
# A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility 

**Authors**: Andreas Hochlehnert, Hardik Bhatnagar, Vishaal Udandarao, Samuel Albanie, Ameya Prabhu, Matthias Bethge  

**Link**: [PDF](https://arxiv.org/pdf/2504.07086)  

**Abstract**: Reasoning has emerged as the next major frontier for language models (LMs), with rapid advances from both academic and industrial labs. However, this progress often outpaces methodological rigor, with many evaluations relying on benchmarking practices that lack transparency, robustness, or statistical grounding. In this work, we conduct a comprehensive empirical study and find that current mathematical reasoning benchmarks are highly sensitive to subtle implementation choices - including decoding parameters, random seeds, prompt formatting, and even hardware and software-framework configurations. Performance gains reported in recent studies frequently hinge on unclear comparisons or unreported sources of variance. To address these issues, we propose a standardized evaluation framework with clearly defined best practices and reporting standards. Using this framework, we reassess recent methods and find that reinforcement learning (RL) approaches yield only modest improvements - far below prior claims - and are prone to overfitting, especially on small-scale benchmarks like AIME24. In contrast, supervised finetuning (SFT) methods show consistently stronger generalization. To foster reproducibility, we release all code, prompts, and model outputs, for reasoning benchmarks, establishing more rigorous foundations for future work. 

---
# SkillWeaver: Web Agents can Self-Improve by Discovering and Honing Skills 

**Authors**: Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, Yu Su  

**Link**: [PDF](https://arxiv.org/pdf/2504.07079)  

**Abstract**: To survive and thrive in complex environments, humans have evolved sophisticated self-improvement mechanisms through environment exploration, hierarchical abstraction of experiences into reuseable skills, and collaborative construction of an ever-growing skill repertoire. Despite recent advancements, autonomous web agents still lack crucial self-improvement capabilities, struggling with procedural knowledge abstraction, refining skills, and skill composition. In this work, we introduce SkillWeaver, a skill-centric framework enabling agents to self-improve by autonomously synthesizing reusable skills as APIs. Given a new website, the agent autonomously discovers skills, executes them for practice, and distills practice experiences into robust APIs. Iterative exploration continually expands a library of lightweight, plug-and-play APIs, significantly enhancing the agent's capabilities. Experiments on WebArena and real-world websites demonstrate the efficacy of SkillWeaver, achieving relative success rate improvements of 31.8% and 39.8%, respectively. Additionally, APIs synthesized by strong agents substantially enhance weaker agents through transferable skills, yielding improvements of up to 54.3% on WebArena. These results demonstrate the effectiveness of honing diverse website interactions into APIs, which can be seamlessly shared among various web agents. 

---
# A Unified Agentic Framework for Evaluating Conditional Image Generation 

**Authors**: Jifang Wang, Xue Yang, Longyue Wang, Zhenran Xu, Yiyu Wang, Yaowei Wang, Weihua Luo, Kaifu Zhang, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07046)  

**Abstract**: Conditional image generation has gained significant attention for its ability to personalize content. However, the field faces challenges in developing task-agnostic, reliable, and explainable evaluation metrics. This paper introduces CIGEval, a unified agentic framework for comprehensive evaluation of conditional image generation tasks. CIGEval utilizes large multimodal models (LMMs) as its core, integrating a multi-functional toolbox and establishing a fine-grained evaluation framework. Additionally, we synthesize evaluation trajectories for fine-tuning, empowering smaller LMMs to autonomously select appropriate tools and conduct nuanced analyses based on tool outputs. Experiments across seven prominent conditional image generation tasks demonstrate that CIGEval (GPT-4o version) achieves a high correlation of 0.4625 with human assessments, closely matching the inter-annotator correlation of 0.47. Moreover, when implemented with 7B open-source LMMs using only 2.3K training trajectories, CIGEval surpasses the previous GPT-4o-based state-of-the-art method. Case studies on GPT-4o image generation highlight CIGEval's capability in identifying subtle issues related to subject consistency and adherence to control guidance, indicating its great potential for automating evaluation of image generation tasks with human-level reliability. 

---
# RNN-Transducer-based Losses for Speech Recognition on Noisy Targets 

**Authors**: Vladimir Bataev  

**Link**: [PDF](https://arxiv.org/pdf/2504.06963)  

**Abstract**: Training speech recognition systems on noisy transcripts is a significant challenge in industrial pipelines, where datasets are enormous and ensuring accurate transcription for every instance is difficult. In this work, we introduce novel loss functions to mitigate the impact of transcription errors in RNN-Transducer models. Our Star-Transducer loss addresses deletion errors by incorporating "skip frame" transitions in the loss lattice, restoring over 90% of the system's performance compared to models trained with accurate transcripts. The Bypass-Transducer loss uses "skip token" transitions to tackle insertion errors, recovering more than 60% of the quality. Finally, the Target-Robust Transducer loss merges these approaches, offering robust performance against arbitrary errors. Experimental results demonstrate that the Target-Robust Transducer loss significantly improves RNN-T performance on noisy data by restoring over 70% of the quality compared to well-transcribed data. 

---
# Adaptive Computation Pruning for the Forgetting Transformer 

**Authors**: Zhixuan Lin, Johan Obando-Ceron, Xu Owen He, Aaron Courville  

**Link**: [PDF](https://arxiv.org/pdf/2504.06949)  

**Abstract**: The recently proposed Forgetting Transformer (FoX) incorporates a forget gate into softmax attention and has shown consistently better or on-par performance compared to the standard RoPE-based Transformer. Notably, many attention heads in FoX tend to forget quickly, causing their output at each timestep to rely primarily on the local context. Based on this observation, we propose Adaptive Computation Pruning (ACP) for FoX, a method that dynamically prunes computations involving input-output dependencies that are strongly decayed by the forget gate. This is achieved using a dynamically set pruning threshold that ensures that the pruned attention weights remain negligible. We apply ACP to language model pretraining with FoX and show it consistently reduces the number of FLOPs in softmax attention by around 70% across different model sizes and context lengths, resulting in a roughly 10% to 35% improvement in training throughput. Furthermore, longer context lengths yield greater computational savings. All these speed improvements are achieved without any performance degradation. We also perform several analyses to provide deeper insights into our method, such as examining the pruning patterns and analyzing the distribution of FLOP savings across different attention heads. Our code is available at this https URL. 

---
# FamilyTool: A Multi-hop Personalized Tool Use Benchmark 

**Authors**: Yuxin Wang, Yiran Guo, Yining Zheng, Zhangyue Yin, Shuo Chen, Jie Yang, Jiajun Chen, Xuanjing Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06766)  

**Abstract**: The integration of tool learning with Large Language Models (LLMs) has expanded their capabilities in handling complex tasks by leveraging external tools. However, existing benchmarks for tool learning inadequately address critical real-world personalized scenarios, particularly those requiring multi-hop reasoning and inductive knowledge adaptation in dynamic environments. To bridge this gap, we introduce FamilyTool, a novel benchmark grounded in a family-based knowledge graph (KG) that simulates personalized, multi-hop tool use scenarios. FamilyTool challenges LLMs with queries spanning 1 to 3 relational hops (e.g., inferring familial connections and preferences) and incorporates an inductive KG setting where models must adapt to unseen user preferences and relationships without re-training, a common limitation in prior approaches that compromises generalization. We further propose KGETool: a simple KG-augmented evaluation pipeline to systematically assess LLMs' tool use ability in these settings. Experiments reveal significant performance gaps in state-of-the-art LLMs, with accuracy dropping sharply as hop complexity increases and inductive scenarios exposing severe generalization deficits. These findings underscore the limitations of current LLMs in handling personalized, evolving real-world contexts and highlight the urgent need for advancements in tool-learning frameworks. FamilyTool serves as a critical resource for evaluating and advancing LLM agents' reasoning, adaptability, and scalability in complex, dynamic environments. Code and dataset are available at Github. 

---
# CAT: Circular-Convolutional Attention for Sub-Quadratic Transformers 

**Authors**: Yoshihiro Yamada  

**Link**: [PDF](https://arxiv.org/pdf/2504.06704)  

**Abstract**: Transformers have driven remarkable breakthroughs in natural language processing and computer vision, yet their standard attention mechanism still imposes O(N^2) complexity, hindering scalability to longer sequences. We introduce Circular-convolutional ATtention (CAT), a Fourier-based approach that efficiently applies circular convolutions to reduce complexity without sacrificing representational power. CAT achieves O(NlogN) computations, requires fewer learnable parameters by streamlining fully-connected layers, and introduces no heavier operations, resulting in consistent accuracy improvements and about a 10% speedup in naive PyTorch implementations on large-scale benchmarks such as ImageNet-1k and WikiText-103. Grounded in an engineering-isomorphism framework, CAT's design not only offers practical efficiency and ease of implementation but also provides insights to guide the development of next-generation, high-performance Transformer architectures. Finally, our ablation studies highlight the key conditions underlying CAT's success, shedding light on broader principles for scalable attention mechanisms. 

---
# Bridging the Gap Between Preference Alignment and Machine Unlearning 

**Authors**: Xiaohua Feng, Yuyuan Li, Huwei Ji, Jiaming Zhang, Li Zhang, Tianyu Du, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06659)  

**Abstract**: Despite advances in Preference Alignment (PA) for Large Language Models (LLMs), mainstream methods like Reinforcement Learning with Human Feedback (RLHF) face notable challenges. These approaches require high-quality datasets of positive preference examples, which are costly to obtain and computationally intensive due to training instability, limiting their use in low-resource scenarios. LLM unlearning technique presents a promising alternative, by directly removing the influence of negative examples. However, current research has primarily focused on empirical validation, lacking systematic quantitative analysis. To bridge this gap, we propose a framework to explore the relationship between PA and LLM unlearning. Specifically, we introduce a bi-level optimization-based method to quantify the impact of unlearning specific negative examples on PA performance. Our analysis reveals that not all negative examples contribute equally to alignment improvement when unlearned, and the effect varies significantly across examples. Building on this insight, we pose a crucial question: how can we optimally select and weight negative examples for unlearning to maximize PA performance? To answer this, we propose a framework called Unlearning to Align (U2A), which leverages bi-level optimization to efficiently select and unlearn examples for optimal PA performance. We validate the proposed method through extensive experiments, with results confirming its effectiveness. 

---
# A Neuro-inspired Interpretation of Unlearning in Large Language Models through Sample-level Unlearning Difficulty 

**Authors**: Xiaohua Feng, Yuyuan Li, Chengye Wang, Junlin Liu, Li Zhang, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06658)  

**Abstract**: Driven by privacy protection laws and regulations, unlearning in Large Language Models (LLMs) is gaining increasing attention. However, current research often neglects the interpretability of the unlearning process, particularly concerning sample-level unlearning difficulty. Existing studies typically assume a uniform unlearning difficulty across samples. This simplification risks attributing the performance of unlearning algorithms to sample selection rather than the algorithm's design, potentially steering the development of LLM unlearning in the wrong direction. Thus, we investigate the relationship between LLM unlearning and sample characteristics, with a focus on unlearning difficulty. Drawing inspiration from neuroscience, we propose a Memory Removal Difficulty ($\mathrm{MRD}$) metric to quantify sample-level unlearning difficulty. Using $\mathrm{MRD}$, we analyze the characteristics of hard-to-unlearn versus easy-to-unlearn samples. Furthermore, we propose an $\mathrm{MRD}$-based weighted sampling method to optimize existing unlearning algorithms, which prioritizes easily forgettable samples, thereby improving unlearning efficiency and effectiveness. We validate the proposed metric and method using public benchmarks and datasets, with results confirming its effectiveness. 

---
# Wanting to be Understood 

**Authors**: Chrisantha Fernando, Dylan Banarse, Simon Osindero  

**Link**: [PDF](https://arxiv.org/pdf/2504.06611)  

**Abstract**: This paper explores an intrinsic motivation for mutual awareness, hypothesizing that humans possess a fundamental drive to understand \textit{and to be understood} even in the absence of extrinsic rewards. Through simulations of the perceptual crossing paradigm, we explore the effect of various internal reward functions in reinforcement learning agents. The drive to understand is implemented as an active inference type artificial curiosity reward, whereas the drive to be understood is implemented through intrinsic rewards for imitation, influence/impressionability, and sub-reaction time anticipation of the other. Results indicate that while artificial curiosity alone does not lead to a preference for social interaction, rewards emphasizing reciprocal understanding successfully drive agents to prioritize interaction. We demonstrate that this intrinsic motivation can facilitate cooperation in tasks where only one agent receives extrinsic reward for the behaviour of the other. 

---
# Defending LLM Watermarking Against Spoofing Attacks with Contrastive Representation Learning 

**Authors**: Li An, Yujian Liu, Yepeng Liu, Yang Zhang, Yuheng Bu, Shiyu Chang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06575)  

**Abstract**: Watermarking has emerged as a promising technique for detecting texts generated by LLMs. Current research has primarily focused on three design criteria: high quality of the watermarked text, high detectability, and robustness against removal attack. However, the security against spoofing attacks remains relatively understudied. For example, a piggyback attack can maliciously alter the meaning of watermarked text-transforming it into hate speech-while preserving the original watermark, thereby damaging the reputation of the LLM provider. We identify two core challenges that make defending against spoofing difficult: (1) the need for watermarks to be both sensitive to semantic-distorting changes and insensitive to semantic-preserving edits, and (2) the contradiction between the need to detect global semantic shifts and the local, auto-regressive nature of most watermarking schemes. To address these challenges, we propose a semantic-aware watermarking algorithm that post-hoc embeds watermarks into a given target text while preserving its original meaning. Our method introduces a semantic mapping model, which guides the generation of a green-red token list, contrastively trained to be sensitive to semantic-distorting changes and insensitive to semantic-preserving changes. Experiments on two standard benchmarks demonstrate strong robustness against removal attacks and security against spoofing attacks, including sentiment reversal and toxic content insertion, while maintaining high watermark detectability. Our approach offers a significant step toward more secure and semantically aware watermarking for LLMs. Our code is available at this https URL. 

---
# Missing Premise exacerbates Overthinking: Are Reasoning Models losing Critical Thinking Skill? 

**Authors**: Chenrui Fan, Ming Li, Lichao Sun, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.06514)  

**Abstract**: We find that the response length of reasoning LLMs, whether trained by reinforcement learning or supervised learning, drastically increases for ill-posed questions with missing premises (MiP), ending up with redundant and ineffective thinking. This newly introduced scenario exacerbates the general overthinking issue to a large extent, which we name as the MiP-Overthinking. Such failures are against the ``test-time scaling law'' but have been widely observed on multiple datasets we curated with MiP, indicating the harm of cheap overthinking and a lack of critical thinking. Surprisingly, LLMs not specifically trained for reasoning exhibit much better performance on the MiP scenario, producing much shorter responses that quickly identify ill-posed queries. This implies a critical flaw of the current training recipe for reasoning LLMs, which does not encourage efficient thinking adequately, leading to the abuse of thinking patterns. To further investigate the reasons behind such failures, we conduct fine-grained analyses of the reasoning length, overthinking patterns, and location of critical thinking on different types of LLMs. Moreover, our extended ablation study reveals that the overthinking is contagious through the distillation of reasoning models' responses. These results improve the understanding of overthinking and shed novel insights into mitigating the problem. 

---
# Understanding Machine Unlearning Through the Lens of Mode Connectivity 

**Authors**: Jiali Cheng, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2504.06407)  

**Abstract**: Machine Unlearning aims to remove undesired information from trained models without requiring full retraining from scratch. Despite recent advancements, their underlying loss landscapes and optimization dynamics received less attention. In this paper, we investigate and analyze machine unlearning through the lens of mode connectivity - the phenomenon where independently trained models can be connected by smooth low-loss paths in the parameter space. We define and study mode connectivity in unlearning across a range of overlooked conditions, including connections between different unlearning methods, models trained with and without curriculum learning, and models optimized with first-order and secondorder techniques. Our findings show distinct patterns of fluctuation of different evaluation metrics along the curve, as well as the mechanistic (dis)similarity between unlearning methods. To the best of our knowledge, this is the first study on mode connectivity in the context of machine unlearning. 

---
# On the Effectiveness and Generalization of Race Representations for Debiasing High-Stakes Decisions 

**Authors**: Dang Nguyen, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.06303)  

**Abstract**: Understanding and mitigating biases is critical for the adoption of large language models (LLMs) in high-stakes decision-making. We introduce Admissions and Hiring, decision tasks with hypothetical applicant profiles where a person's race can be inferred from their name, as simplified test beds for racial bias. We show that Gemma 2B Instruct and LLaMA 3.2 3B Instruct exhibit strong biases. Gemma grants admission to 26% more White than Black applicants, and LLaMA hires 60% more Asian than White applicants. We demonstrate that these biases are resistant to prompt engineering: multiple prompting strategies all fail to promote fairness. In contrast, using distributed alignment search, we can identify "race subspaces" within model activations and intervene on them to debias model decisions. Averaging the representation across all races within the subspaces reduces Gemma's bias by 37-57%. Finally, we examine the generalizability of Gemma's race subspaces, and find limited evidence for generalization, where changing the prompt format can affect the race representation. Our work suggests mechanistic approaches may provide a promising venue for improving the fairness of LLMs, but a universal race representation remains elusive. 

---
# A Diverse and Effective Retrieval-Based Debt Collection System with Expert Knowledge 

**Authors**: Jiaming Luo, Weiyi Luo, Guoqing Sun, Mengchen Zhu, Haifeng Tang, Kunyao Lan, Mengyue Wu, Kenny Q. Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06273)  

**Abstract**: Designing effective debt collection systems is crucial for improving operational efficiency and reducing costs in the financial industry. However, the challenges of maintaining script diversity, contextual relevance, and coherence make this task particularly difficult. This paper presents a debt collection system based on real debtor-collector data from a major commercial bank. We construct a script library from real-world debt collection conversations, and propose a two-stage retrieval based response system for contextual relevance. Experimental results show that our system improves script diversity, enhances response relevance, and achieves practical deployment efficiency through knowledge distillation. This work offers a scalable and automated solution, providing valuable insights for advancing debt collection practices in real-world applications. 

---
# ER-RAG: Enhance RAG with ER-Based Unified Modeling of Heterogeneous Data Sources 

**Authors**: Yikuan Xia, Jiazun Chen, Yirui Zhan, Suifeng Zhao, Weipeng Jiang, Chaorui Zhang, Wei Han, Bo Bai, Jun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.06271)  

**Abstract**: Large language models (LLMs) excel in question-answering (QA) tasks, and retrieval-augmented generation (RAG) enhances their precision by incorporating external evidence from diverse sources like web pages, databases, and knowledge graphs. However, current RAG methods rely on agent-specific strategies for individual data sources, posing challenges low-resource or black-box environments and complicates operations when evidence is fragmented across sources. To address these limitations, we propose ER-RAG, a framework that unifies evidence integration across heterogeneous data sources using the Entity-Relationship (ER) model. ER-RAG standardizes entity retrieval and relationship querying through ER-based APIs with GET and JOIN operations. It employs a two-stage generation process: first, a preference optimization module selects optimal sources; second, another module constructs API chains based on source schemas. This unified approach allows efficient fine-tuning and seamless integration across diverse data sources. ER-RAG demonstrated its effectiveness by winning all three tracks of the 2024 KDDCup CRAG Challenge, achieving performance on par with commercial RAG pipelines using an 8B LLM backbone. It outperformed hybrid competitors by 3.1% in LLM score and accelerated retrieval by 5.5X. 

---
# EXCLAIM: An Explainable Cross-Modal Agentic System for Misinformation Detection with Hierarchical Retrieval 

**Authors**: Yin Wu, Zhengxuan Zhang, Fuling Wang, Yuyu Luo, Hui Xiong, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06269)  

**Abstract**: Misinformation continues to pose a significant challenge in today's information ecosystem, profoundly shaping public perception and behavior. Among its various manifestations, Out-of-Context (OOC) misinformation is particularly obscure, as it distorts meaning by pairing authentic images with misleading textual narratives. Existing methods for detecting OOC misinformation predominantly rely on coarse-grained similarity metrics between image-text pairs, which often fail to capture subtle inconsistencies or provide meaningful explainability. While multi-modal large language models (MLLMs) demonstrate remarkable capabilities in visual reasoning and explanation generation, they have not yet demonstrated the capacity to address complex, fine-grained, and cross-modal distinctions necessary for robust OOC detection. To overcome these limitations, we introduce EXCLAIM, a retrieval-based framework designed to leverage external knowledge through multi-granularity index of multi-modal events and entities. Our approach integrates multi-granularity contextual analysis with a multi-agent reasoning architecture to systematically evaluate the consistency and integrity of multi-modal news content. Comprehensive experiments validate the effectiveness and resilience of EXCLAIM, demonstrating its ability to detect OOC misinformation with 4.3% higher accuracy compared to state-of-the-art approaches, while offering explainable and actionable insights. 

---
# Information-Theoretic Reward Decomposition for Generalizable RLHF 

**Authors**: Liyuan Mao, Haoran Xu, Amy Zhang, Weinan Zhang, Chenjia Bai  

**Link**: [PDF](https://arxiv.org/pdf/2504.06020)  

**Abstract**: A generalizable reward model is crucial in Reinforcement Learning from Human Feedback (RLHF) as it enables correctly evaluating unseen prompt-response pairs. However, existing reward models lack this ability, as they are typically trained by increasing the reward gap between chosen and rejected responses, while overlooking the prompts that the responses are conditioned on. Consequently, when the trained reward model is evaluated on prompt-response pairs that lie outside the data distribution, neglecting the effect of prompts may result in poor generalization of the reward model. To address this issue, we decompose the reward value into two independent components: prompt-free reward and prompt-related reward. Prompt-free reward represents the evaluation that is determined only by responses, while the prompt-related reward reflects the reward that derives from both the prompt and the response. We extract these two components from an information-theoretic perspective, which requires no extra models. Subsequently, we propose a new reward learning algorithm by prioritizing data samples based on their prompt-free reward values. Through toy examples, we demonstrate that the extracted prompt-free and prompt-related rewards effectively characterize two parts of the reward model. Further, standard evaluations show that our method improves both the alignment performance and the generalization capability of the reward model. 

---
# StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization 

**Authors**: Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue Zhao, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05804)  

**Abstract**: The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present StealthRank, a novel adversarial ranking attack that manipulates LLM-driven product recommendation systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within product descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target products while avoiding explicit manipulation traces that can be easily detected. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven recommendation systems. 

---
# MultiDelete for Multimodal Machine Unlearning 

**Authors**: Jiali Cheng, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2311.12047)  

**Abstract**: Machine Unlearning removes specific knowledge about training data samples from an already trained model. It has significant practical benefits, such as purging private, inaccurate, or outdated information from trained models without the need for complete re-training. Unlearning within a multimodal setting presents unique challenges due to the complex dependencies between different data modalities and the expensive cost of training on large multimodal datasets and architectures. This paper presents the first machine unlearning approach for multimodal data and models, titled MultiDelete, which is designed to decouple associations between unimodal data points during unlearning without losing the overall representation strength of the trained model. MultiDelete advocates for three key properties for effective multimodal unlearning: (a): modality decoupling, which effectively decouples the association between individual unimodal data points marked for deletion, rendering them as unrelated data points, (b): multimodal knowledge retention, which retains the multimodal representation post-unlearning, and (c): unimodal knowledge retention, which retains the unimodal representation postunlearning. MultiDelete is efficient to train and is not constrained by using a strongly convex loss -- a common restriction among existing baselines. Experiments on two architectures and four datasets, including image-text and graph-text datasets, show that MultiDelete gains an average improvement of 17.6 points over best performing baseline in unlearning multimodal samples, can maintain the multimodal and unimodal knowledge of the original model post unlearning, and can provide better protection to unlearned data against adversarial attacks. 

---
