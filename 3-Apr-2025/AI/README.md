# The LLM Wears Prada: Analysing Gender Bias and Stereotypes through Online Shopping Data 

**Authors**: Massimiliano Luca, Ciro Beneduce, Bruno Lepri, Jacopo Staiano  

**Link**: [PDF](https://arxiv.org/pdf/2504.01951)  

**Abstract**: With the wide and cross-domain adoption of Large Language Models, it becomes crucial to assess to which extent the statistical correlations in training data, which underlie their impressive performance, hide subtle and potentially troubling biases. Gender bias in LLMs has been widely investigated from the perspectives of works, hobbies, and emotions typically associated with a specific gender. In this study, we introduce a novel perspective. We investigate whether LLMs can predict an individual's gender based solely on online shopping histories and whether these predictions are influenced by gender biases and stereotypes. Using a dataset of historical online purchases from users in the United States, we evaluate the ability of six LLMs to classify gender and we then analyze their reasoning and products-gender co-occurrences. Results indicate that while models can infer gender with moderate accuracy, their decisions are often rooted in stereotypical associations between product categories and gender. Furthermore, explicit instructions to avoid bias reduce the certainty of model predictions, but do not eliminate stereotypical patterns. Our findings highlight the persistent nature of gender biases in LLMs and emphasize the need for robust bias-mitigation strategies. 

---
# Critical Thinking: Which Kinds of Complexity Govern Optimal Reasoning Length? 

**Authors**: Celine Lee, Alexander M. Rush, Keyon Vafa  

**Link**: [PDF](https://arxiv.org/pdf/2504.01935)  

**Abstract**: Large language models (LLMs) often benefit from verbalized reasoning at inference time, but it remains unclear which aspects of task difficulty these extra reasoning tokens address. To investigate this question, we formalize a framework using deterministic finite automata (DFAs). DFAs offer a formalism through which we can characterize task complexity through measurable properties such as run length (number of reasoning steps required) and state-space size (decision complexity). We first show that across different tasks and models of different sizes and training paradigms, there exists an optimal amount of reasoning tokens such that the probability of producing a correct solution is maximized. We then investigate which properties of complexity govern this critical length: we find that task instances with longer corresponding underlying DFA runs (i.e. demand greater latent state-tracking requirements) correlate with longer reasoning lengths, but, surprisingly, that DFA size (i.e. state-space complexity) does not. We then demonstrate an implication of these findings: being able to predict the optimal number of reasoning tokens for new problems and filtering out non-optimal length answers results in consistent accuracy improvements. 

---
# Advancing AI-Scientist Understanding: Making LLM Think Like a Physicist with Interpretable Reasoning 

**Authors**: Yinggan Xu, Hana Kimlee, Yijia Xiao, Di Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.01911)  

**Abstract**: Large Language Models (LLMs) are playing an expanding role in physics research by enhancing reasoning, symbolic manipulation, and numerical computation. However, ensuring the reliability and interpretability of their outputs remains a significant challenge. In our framework, we conceptualize the collaboration between AI and human scientists as a dynamic interplay among three modules: the reasoning module, the interpretation module, and the AI-scientist interaction module. Recognizing that effective physics reasoning demands rigorous logical consistency, quantitative precision, and deep integration with established theoretical models, we introduce the interpretation module to improve the understanding of AI-generated outputs, which is not previously explored in the literature. This module comprises multiple specialized agents, including summarizers, model builders, UI builders, and testers, which collaboratively structure LLM outputs within a physically grounded framework, by constructing a more interpretable science model. A case study demonstrates that our approach enhances transparency, facilitates validation, and strengthens AI-augmented reasoning in scientific discovery. 

---
# CoRAG: Collaborative Retrieval-Augmented Generation 

**Authors**: Aashiq Muhamed, Mona Diab, Virginia Smith  

**Link**: [PDF](https://arxiv.org/pdf/2504.01883)  

**Abstract**: Retrieval-Augmented Generation (RAG) models excel in knowledge-intensive tasks, especially under few-shot learning constraints. We introduce CoRAG, a framework extending RAG to collaborative settings, where clients jointly train a shared model using a collaborative passage store. To evaluate CoRAG, we introduce CRAB, a benchmark for collaborative homogeneous open-domain question answering. Our experiments demonstrate that CoRAG consistently outperforms both parametric collaborative learning methods and locally trained RAG models in low-resource scenarios. Further analysis reveals the critical importance of relevant passages within the shared store, the surprising benefits of incorporating irrelevant passages, and the potential for hard negatives to negatively impact performance. This introduces a novel consideration in collaborative RAG: the trade-off between leveraging a collectively enriched knowledge base and the potential risk of incorporating detrimental passages from other clients. Our findings underscore the viability of CoRAG, while also highlighting key design challenges and promising avenues for future research. 

---
# An Approach to Technical AGI Safety and Security 

**Authors**: Rohin Shah, Alex Irpan, Alexander Matt Turner, Anna Wang, Arthur Conmy, David Lindner, Jonah Brown-Cohen, Lewis Ho, Neel Nanda, Raluca Ada Popa, Rishub Jain, Rory Greig, Samuel Albanie, Scott Emmons, Sebastian Farquhar, Sébastien Krier, Senthooran Rajamanoharan, Sophie Bridgers, Tobi Ijitoye, Tom Everitt, Victoria Krakovna, Vikrant Varma, Vladimir Mikulik, Zachary Kenton, Dave Orr, Shane Legg, Noah Goodman, Allan Dafoe, Four Flynn, Anca Dragan  

**Link**: [PDF](https://arxiv.org/pdf/2504.01849)  

**Abstract**: Artificial General Intelligence (AGI) promises transformative benefits but also presents significant risks. We develop an approach to address the risk of harms consequential enough to significantly harm humanity. We identify four areas of risk: misuse, misalignment, mistakes, and structural risks. Of these, we focus on technical approaches to misuse and misalignment. For misuse, our strategy aims to prevent threat actors from accessing dangerous capabilities, by proactively identifying dangerous capabilities, and implementing robust security, access restrictions, monitoring, and model safety mitigations. To address misalignment, we outline two lines of defense. First, model-level mitigations such as amplified oversight and robust training can help to build an aligned model. Second, system-level security measures such as monitoring and access control can mitigate harm even if the model is misaligned. Techniques from interpretability, uncertainty estimation, and safer design patterns can enhance the effectiveness of these mitigations. Finally, we briefly outline how these ingredients could be combined to produce safety cases for AGI systems. 

---
# PaperBench: Evaluating AI's Ability to Replicate AI Research 

**Authors**: Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, Johannes Heidecke, Amelia Glaese, Tejal Patwardhan  

**Link**: [PDF](https://arxiv.org/pdf/2504.01848)  

**Abstract**: We introduce PaperBench, a benchmark evaluating the ability of AI agents to replicate state-of-the-art AI research. Agents must replicate 20 ICML 2024 Spotlight and Oral papers from scratch, including understanding paper contributions, developing a codebase, and successfully executing experiments. For objective evaluation, we develop rubrics that hierarchically decompose each replication task into smaller sub-tasks with clear grading criteria. In total, PaperBench contains 8,316 individually gradable tasks. Rubrics are co-developed with the author(s) of each ICML paper for accuracy and realism. To enable scalable evaluation, we also develop an LLM-based judge to automatically grade replication attempts against rubrics, and assess our judge's performance by creating a separate benchmark for judges. We evaluate several frontier models on PaperBench, finding that the best-performing tested agent, Claude 3.5 Sonnet (New) with open-source scaffolding, achieves an average replication score of 21.0\%. Finally, we recruit top ML PhDs to attempt a subset of PaperBench, finding that models do not yet outperform the human baseline. We \href{this https URL}{open-source our code} to facilitate future research in understanding the AI engineering capabilities of AI agents. 

---
# A Novel Approach To Implementing Knowledge Distillation In Tsetlin Machines 

**Authors**: Calvin Kinateder  

**Link**: [PDF](https://arxiv.org/pdf/2504.01798)  

**Abstract**: The Tsetlin Machine (TM) is a propositional logic based model that uses conjunctive clauses to learn patterns from data. As with typical neural networks, the performance of a Tsetlin Machine is largely dependent on its parameter count, with a larger number of parameters producing higher accuracy but slower execution. Knowledge distillation in neural networks transfers information from an already-trained teacher model to a smaller student model to increase accuracy in the student without increasing execution time. We propose a novel approach to implementing knowledge distillation in Tsetlin Machines by utilizing the probability distributions of each output sample in the teacher to provide additional context to the student. Additionally, we propose a novel clause-transfer algorithm that weighs the importance of each clause in the teacher and initializes the student with only the most essential data. We find that our algorithm can significantly improve performance in the student model without negatively impacting latency in the tested domains of image recognition and text classification. 

---
# Enhancing Interpretability in Generative AI Through Search-Based Data Influence Analysis 

**Authors**: Theodoros Aivalis, Iraklis A. Klampanos, Antonis Troumpoukis, Joemon M. Jose  

**Link**: [PDF](https://arxiv.org/pdf/2504.01771)  

**Abstract**: Generative AI models offer powerful capabilities but often lack transparency, making it difficult to interpret their output. This is critical in cases involving artistic or copyrighted content. This work introduces a search-inspired approach to improve the interpretability of these models by analysing the influence of training data on their outputs. Our method provides observational interpretability by focusing on a model's output rather than on its internal state. We consider both raw data and latent-space embeddings when searching for the influence of data items in generated content. We evaluate our method by retraining models locally and by demonstrating the method's ability to uncover influential subsets in the training data. This work lays the groundwork for future extensions, including user-based evaluations with domain experts, which is expected to improve observational interpretability further. 

---
# Epistemic Skills: Reasoning about Knowledge and Oblivion 

**Authors**: Xiaolong Liang, Yì N. Wáng  

**Link**: [PDF](https://arxiv.org/pdf/2504.01733)  

**Abstract**: This paper presents a class of epistemic logics that captures the dynamics of acquiring knowledge and descending into oblivion, while incorporating concepts of group knowledge. The approach is grounded in a system of weighted models, introducing an ``epistemic skills'' metric to represent the epistemic capacities tied to knowledge updates. Within this framework, knowledge acquisition is modeled as a process of upskilling, whereas oblivion is represented as a consequence of downskilling. The framework further enables exploration of ``knowability'' and ``forgettability,'' defined as the potential to gain knowledge through upskilling and to lapse into oblivion through downskilling, respectively. Additionally, it supports a detailed analysis of the distinctions between epistemic de re and de dicto expressions. The computational complexity of the model checking and satisfiability problems is examined, offering insights into their theoretical foundations and practical implications. 

---
# Proposition of Affordance-Driven Environment Recognition Framework Using Symbol Networks in Large Language Models 

**Authors**: Kazuma Arii, Satoshi Kurihara  

**Link**: [PDF](https://arxiv.org/pdf/2504.01644)  

**Abstract**: In the quest to enable robots to coexist with humans, understanding dynamic situations and selecting appropriate actions based on common sense and affordances are essential. Conventional AI systems face challenges in applying affordance, as it represents implicit knowledge derived from common sense. However, large language models (LLMs) offer new opportunities due to their ability to process extensive human knowledge. This study proposes a method for automatic affordance acquisition by leveraging LLM outputs. The process involves generating text using LLMs, reconstructing the output into a symbol network using morphological and dependency analysis, and calculating affordances based on network distances. Experiments using ``apple'' as an example demonstrated the method's ability to extract context-dependent affordances with high explainability. The results suggest that the proposed symbol network, reconstructed from LLM outputs, enables robots to interpret affordances effectively, bridging the gap between symbolized data and human-like situational understanding. 

---
# LLM-mediated Dynamic Plan Generation with a Multi-Agent Approach 

**Authors**: Reo Abe, Akifumi Ito, Kanata Takayasu, Satoshi Kurihara  

**Link**: [PDF](https://arxiv.org/pdf/2504.01637)  

**Abstract**: Planning methods with high adaptability to dynamic environments are crucial for the development of autonomous and versatile robots. We propose a method for leveraging a large language model (GPT-4o) to automatically generate networks capable of adapting to dynamic environments. The proposed method collects environmental "status," representing conditions and goals, and uses them to generate agents. These agents are interconnected on the basis of specific conditions, resulting in networks that combine flexibility and generality. We conducted evaluation experiments to compare the networks automatically generated with the proposed method with manually constructed ones, confirming the comprehensiveness of the proposed method's networks and their higher generality. This research marks a significant advancement toward the development of versatile planning methods applicable to robotics, autonomous vehicles, smart systems, and other complex environments. 

---
# Identifying Macro Causal Effects in C-DMGs 

**Authors**: Simon Ferreira, Charles K. Assaad  

**Link**: [PDF](https://arxiv.org/pdf/2504.01551)  

**Abstract**: Causal effect identification using causal graphs is a fundamental challenge in causal inference. While extensive research has been conducted in this area, most existing methods assume the availability of fully specified causal graphs. However, in complex domains such as medicine and epidemiology, complete causal knowledge is often unavailable, and only partial information about the system is accessible. This paper focuses on causal effect identification within partially specified causal graphs, with particular emphasis on cluster-directed mixed graphs (C-DMGs). These graphs provide a higher-level representation of causal relationships by grouping variables into clusters, offering a more practical approach for handling complex systems. Unlike fully specified causal graphs, C-DMGs can contain cycles, which complicate their analysis and interpretation. Furthermore, their cluster-based nature introduces new challenges, as it gives rise to two distinct types of causal effects, macro causal effects and micro causal effects, with different properties. In this work, we focus on macro causal effects, which describe the effects of entire clusters on other clusters. We establish that the do-calculus is both sound and complete for identifying these effects in C-DMGs. Additionally, we provide a graphical characterization of non-identifiability for macro causal effects in these graphs. 

---
# AI-Newton: A Concept-Driven Physical Law Discovery System without Prior Physical Knowledge 

**Authors**: You-Le Fang, Dong-Shan Jian, Xiang Li, Yan-Qing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.01538)  

**Abstract**: Current limitations in human scientific discovery necessitate a new research paradigm. While advances in artificial intelligence (AI) offer a highly promising solution, enabling AI to emulate human-like scientific discovery remains an open challenge. To address this, we propose AI-Newton, a concept-driven discovery system capable of autonomously deriving physical laws from raw data -- without supervision or prior physical knowledge. The system integrates a knowledge base and knowledge representation centered on physical concepts, along with an autonomous discovery workflow. As a proof of concept, we apply AI-Newton to a large set of Newtonian mechanics problems. Given experimental data with noise, the system successfully rediscovers fundamental laws, including Newton's second law, energy conservation and law of gravitation, using autonomously defined concepts. This achievement marks a significant step toward AI-driven autonomous scientific discovery. 

---
# Enabling Systematic Generalization in Abstract Spatial Reasoning through Meta-Learning for Compositionality 

**Authors**: Philipp Mondorf, Shijia Zhou, Monica Riedler, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2504.01445)  

**Abstract**: Systematic generalization refers to the capacity to understand and generate novel combinations from known components. Despite recent progress by large language models (LLMs) across various domains, these models often fail to extend their knowledge to novel compositional scenarios, revealing notable limitations in systematic generalization. There has been an ongoing debate about whether neural networks possess the capacity for systematic generalization, with recent studies suggesting that meta-learning approaches designed for compositionality can significantly enhance this ability. However, these insights have largely been confined to linguistic problems, leaving their applicability to other tasks an open question. In this study, we extend the approach of meta-learning for compositionality to the domain of abstract spatial reasoning. To this end, we introduce $\textit{SYGAR}$-a dataset designed to evaluate the capacity of models to systematically generalize from known geometric transformations (e.g., translation, rotation) of two-dimensional objects to novel combinations of these transformations (e.g., translation+rotation). Our results show that a transformer-based encoder-decoder model, trained via meta-learning for compositionality, can systematically generalize to previously unseen transformation compositions, significantly outperforming state-of-the-art LLMs, including o3-mini, GPT-4o, and Gemini 2.0 Flash, which fail to exhibit similar systematic behavior. Our findings highlight the effectiveness of meta-learning in promoting systematicity beyond linguistic tasks, suggesting a promising direction toward more robust and generalizable models. 

---
# An Illusion of Progress? Assessing the Current State of Web Agents 

**Authors**: Tianci Xue, Weijian Qi, Tianneng Shi, Chan Hee Song, Boyu Gou, Dawn Song, Huan Sun, Yu Su  

**Link**: [PDF](https://arxiv.org/pdf/2504.01382)  

**Abstract**: As digitalization and cloud technologies evolve, the web is becoming increasingly important in the modern society. Autonomous web agents based on large language models (LLMs) hold a great potential in work automation. It is therefore important to accurately measure and monitor the progression of their capabilities. In this work, we conduct a comprehensive and rigorous assessment of the current state of web agents. Our results depict a very different picture of the competency of current agents, suggesting over-optimism in previously reported results. This gap can be attributed to shortcomings in existing benchmarks. We introduce Online-Mind2Web, an online evaluation benchmark consisting of 300 diverse and realistic tasks spanning 136 websites. It enables us to evaluate web agents under a setting that approximates how real users use these agents. To facilitate more scalable evaluation and development, we also develop a novel LLM-as-a-Judge automatic evaluation method and show that it can achieve around 85% agreement with human judgment, substantially higher than existing methods. Finally, we present the first comprehensive comparative analysis of current web agents, highlighting both their strengths and limitations to inspire future research. 

---
# An Explainable Reconfiguration-Based Optimization Algorithm for Industrial and Reliability-Redundancy Allocation Problems 

**Authors**: Dikshit Chauhan, Nitin Gupta, Anupam Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2504.01331)  

**Abstract**: Industrial and reliability optimization problems often involve complex constraints and require efficient, interpretable solutions. This paper presents AI-AEFA, an advanced parameter reconfiguration-based metaheuristic algorithm designed to address large-scale industrial and reliability-redundancy allocation problems. AI-AEFA enhances search space exploration and convergence efficiency through a novel log-sigmoid-based parameter adaptation and chaotic mapping mechanism. The algorithm is validated across twenty-eight IEEE CEC 2017 constrained benchmark problems, fifteen large-scale industrial optimization problems, and seven reliability-redundancy allocation problems, consistently outperforming state-of-the-art optimization techniques in terms of feasibility, computational efficiency, and convergence speed. The additional key contribution of this work is the integration of SHAP (Shapley Additive Explanations) to enhance the interpretability of AI-AEFA, providing insights into the impact of key parameters such as Coulomb's constant, charge, acceleration, and electrostatic force. This explainability feature enables a deeper understanding of decision-making within the AI-AEFA framework during the optimization processes. The findings confirm AI-AEFA as a robust, scalable, and interpretable optimization tool with significant real-world applications. 

---
# Strategize Globally, Adapt Locally: A Multi-Turn Red Teaming Agent with Dual-Level Learning 

**Authors**: Si Chen, Xiao Yu, Ninareh Mehrabi, Rahul Gupta, Zhou Yu, Ruoxi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2504.01278)  

**Abstract**: The exploitation of large language models (LLMs) for malicious purposes poses significant security risks as these models become more powerful and widespread. While most existing red-teaming frameworks focus on single-turn attacks, real-world adversaries typically operate in multi-turn scenarios, iteratively probing for vulnerabilities and adapting their prompts based on threat model responses. In this paper, we propose \AlgName, a novel multi-turn red-teaming agent that emulates sophisticated human attackers through complementary learning dimensions: global tactic-wise learning that accumulates knowledge over time and generalizes to new attack goals, and local prompt-wise learning that refines implementations for specific goals when initial attempts fail. Unlike previous multi-turn approaches that rely on fixed strategy sets, \AlgName enables the agent to identify new jailbreak tactics, develop a goal-based tactic selection framework, and refine prompt formulations for selected tactics. Empirical evaluations on JailbreakBench demonstrate our framework's superior performance, achieving over 90\% attack success rates against GPT-3.5-Turbo and Llama-3.1-70B within 5 conversation turns, outperforming state-of-the-art baselines. These results highlight the effectiveness of dynamic learning in identifying and exploiting model vulnerabilities in realistic multi-turn scenarios. 

---
# Off-Policy Evaluation for Sequential Persuasion Process with Unobserved Confounding 

**Authors**: Nishanth Venkatesh S., Heeseung Bang, Andreas A. Malikopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2504.01211)  

**Abstract**: In this paper, we expand the Bayesian persuasion framework to account for unobserved confounding variables in sender-receiver interactions. While traditional models assume that belief updates follow Bayesian principles, real-world scenarios often involve hidden variables that impact the receiver's belief formation and decision-making. We conceptualize this as a sequential decision-making problem, where the sender and receiver interact over multiple rounds. In each round, the sender communicates with the receiver, who also interacts with the environment. Crucially, the receiver's belief update is affected by an unobserved confounding variable. By reformulating this scenario as a Partially Observable Markov Decision Process (POMDP), we capture the sender's incomplete information regarding both the dynamics of the receiver's beliefs and the unobserved confounder. We prove that finding an optimal observation-based policy in this POMDP is equivalent to solving for an optimal signaling strategy in the original persuasion framework. Furthermore, we demonstrate how this reformulation facilitates the application of proximal learning for off-policy evaluation in the persuasion process. This advancement enables the sender to evaluate alternative signaling strategies using only observational data from a behavioral policy, thus eliminating the necessity for costly new experiments. 

---
# Remember, but also, Forget: Bridging Myopic and Perfect Recall Fairness with Past-Discounting 

**Authors**: Ashwin Kumar, William Yeoh  

**Link**: [PDF](https://arxiv.org/pdf/2504.01154)  

**Abstract**: Dynamic resource allocation in multi-agent settings often requires balancing efficiency with fairness over time--a challenge inadequately addressed by conventional, myopic fairness measures. Motivated by behavioral insights that human judgments of fairness evolve with temporal distance, we introduce a novel framework for temporal fairness that incorporates past-discounting mechanisms. By applying a tunable discount factor to historical utilities, our approach interpolates between instantaneous and perfect-recall fairness, thereby capturing both immediate outcomes and long-term equity considerations. Beyond aligning more closely with human perceptions of fairness, this past-discounting method ensures that the augmented state space remains bounded, significantly improving computational tractability in sequential decision-making settings. We detail the formulation of discounted-recall fairness in both additive and averaged utility contexts, illustrate its benefits through practical examples, and discuss its implications for designing balanced, scalable resource allocation strategies. 

---
# Efficient Federated Learning Tiny Language Models for Mobile Network Feature Prediction 

**Authors**: Daniel Becking, Ingo Friese, Karsten Müller, Thomas Buchholz, Mandy Galkow-Schneider, Wojciech Samek, Detlev Marpe  

**Link**: [PDF](https://arxiv.org/pdf/2504.01947)  

**Abstract**: In telecommunications, Autonomous Networks (ANs) automatically adjust configurations based on specific requirements (e.g., bandwidth) and available resources. These networks rely on continuous monitoring and intelligent mechanisms for self-optimization, self-repair, and self-protection, nowadays enhanced by Neural Networks (NNs) to enable predictive modeling and pattern recognition. Here, Federated Learning (FL) allows multiple AN cells - each equipped with NNs - to collaboratively train models while preserving data privacy. However, FL requires frequent transmission of large neural data and thus an efficient, standardized compression strategy for reliable communication. To address this, we investigate NNCodec, a Fraunhofer implementation of the ISO/IEC Neural Network Coding (NNC) standard, within a novel FL framework that integrates tiny language models (TLMs) for various mobile network feature prediction (e.g., ping, SNR or band frequency). Our experimental results on the Berlin V2X dataset demonstrate that NNCodec achieves transparent compression (i.e., negligible performance loss) while reducing communication overhead to below 1%, showing the effectiveness of combining NNC with FL in collaboratively learned autonomous mobile networks. 

---
# A thorough benchmark of automatic text classification: From traditional approaches to large language models 

**Authors**: Washington Cunha, Leonardo Rocha, Marcos André Gonçalves  

**Link**: [PDF](https://arxiv.org/pdf/2504.01930)  

**Abstract**: Automatic text classification (ATC) has experienced remarkable advancements in the past decade, best exemplified by recent small and large language models (SLMs and LLMs), leveraged by Transformer architectures. Despite recent effectiveness improvements, a comprehensive cost-benefit analysis investigating whether the effectiveness gains of these recent approaches compensate their much higher costs when compared to more traditional text classification approaches such as SVMs and Logistic Regression is still missing in the literature. In this context, this work's main contributions are twofold: (i) we provide a scientifically sound comparative analysis of the cost-benefit of twelve traditional and recent ATC solutions including five open LLMs, and (ii) a large benchmark comprising {22 datasets}, including sentiment analysis and topic classification, with their (train-validation-test) partitions based on folded cross-validation procedures, along with documentation, and code. The release of code, data, and documentation enables the community to replicate experiments and advance the field in a more scientifically sound manner. Our comparative experimental results indicate that LLMs outperform traditional approaches (up to 26%-7.1% on average) and SLMs (up to 4.9%-1.9% on average) in terms of effectiveness. However, LLMs incur significantly higher computational costs due to fine-tuning, being, on average 590x and 8.5x slower than traditional methods and SLMs, respectively. Results suggests the following recommendations: (1) LLMs for applications that require the best possible effectiveness and can afford the costs; (2) traditional methods such as Logistic Regression and SVM for resource-limited applications or those that cannot afford the cost of tuning large LLMs; and (3) SLMs like Roberta for near-optimal effectiveness-efficiency trade-off. 

---
# Equivariant Spherical CNNs for Accurate Fiber Orientation Distribution Estimation in Neonatal Diffusion MRI with Reduced Acquisition Time 

**Authors**: Haykel Snoussi, Davood Karimi  

**Link**: [PDF](https://arxiv.org/pdf/2504.01925)  

**Abstract**: Early and accurate assessment of brain microstructure using diffusion Magnetic Resonance Imaging (dMRI) is crucial for identifying neurodevelopmental disorders in neonates, but remains challenging due to low signal-to-noise ratio (SNR), motion artifacts, and ongoing myelination. In this study, we propose a rotationally equivariant Spherical Convolutional Neural Network (sCNN) framework tailored for neonatal dMRI. We predict the Fiber Orientation Distribution (FOD) from multi-shell dMRI signals acquired with a reduced set of gradient directions (30% of the full protocol), enabling faster and more cost-effective acquisitions. We train and evaluate the performance of our sCNN using real data from 43 neonatal dMRI datasets provided by the Developing Human Connectome Project (dHCP). Our results demonstrate that the sCNN achieves significantly lower mean squared error (MSE) and higher angular correlation coefficient (ACC) compared to a Multi-Layer Perceptron (MLP) baseline, indicating improved accuracy in FOD estimation. Furthermore, tractography results based on the sCNN-predicted FODs show improved anatomical plausibility, coverage, and coherence compared to those from the MLP. These findings highlight that sCNNs, with their inherent rotational equivariance, offer a promising approach for accurate and clinically efficient dMRI analysis, paving the way for improved diagnostic capabilities and characterization of early brain development. 

---
# Bridging the Linguistic Divide: A Survey on Leveraging Large Language Models for Machine Translation 

**Authors**: Baban Gain, Dibyanayan Bandyopadhyay, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2504.01919)  

**Abstract**: The advent of Large Language Models (LLMs) has significantly reshaped the landscape of machine translation (MT), particularly for low-resource languages and domains that lack sufficient parallel corpora, linguistic tools, and computational infrastructure. This survey presents a comprehensive overview of recent progress in leveraging LLMs for MT. We analyze techniques such as few-shot prompting, cross-lingual transfer, and parameter-efficient fine-tuning that enable effective adaptation to under-resourced settings. The paper also explores synthetic data generation strategies using LLMs, including back-translation and lexical augmentation. Additionally, we compare LLM-based translation with traditional encoder-decoder models across diverse language pairs, highlighting the strengths and limitations of each. We discuss persistent challenges such as hallucinations, evaluation inconsistencies, and inherited biases while also evaluating emerging LLM-driven metrics for translation quality. This survey offers practical insights and outlines future directions for building robust, inclusive, and scalable MT systems in the era of large-scale generative models. 

---
# FineLIP: Extending CLIP's Reach via Fine-Grained Alignment with Longer Text Inputs 

**Authors**: Mothilal Asokan, Kebin Wu, Fatima Albreiki  

**Link**: [PDF](https://arxiv.org/pdf/2504.01916)  

**Abstract**: As a pioneering vision-language model, CLIP (Contrastive Language-Image Pre-training) has achieved significant success across various domains and a wide range of downstream vision-language tasks. However, the text encoders in popular CLIP models are limited to processing only 77 text tokens, which constrains their ability to effectively handle longer, detail-rich captions. Additionally, CLIP models often struggle to effectively capture detailed visual and textual information, which hampers their performance on tasks that require fine-grained analysis. To address these limitations, we present a novel approach, \textbf{FineLIP}, that extends the capabilities of CLIP. FineLIP enhances cross-modal text-image mapping by incorporating \textbf{Fine}-grained alignment with \textbf{L}onger text input within the CL\textbf{IP}-style framework. FineLIP first extends the positional embeddings to handle longer text, followed by the dynamic aggregation of local image and text tokens. The aggregated results are then used to enforce fine-grained token-to-token cross-modal alignment. We validate our model on datasets with long, detailed captions across two tasks: zero-shot cross-modal retrieval and text-to-image generation. Quantitative and qualitative experimental results demonstrate the effectiveness of FineLIP, outperforming existing state-of-the-art approaches. Furthermore, comprehensive ablation studies validate the benefits of key design elements within FineLIP. 

---
# Benchmarking Synthetic Tabular Data: A Multi-Dimensional Evaluation Framework 

**Authors**: Andrey Sidorenko, Michael Platzer, Mario Scriminaci, Paul Tiwald  

**Link**: [PDF](https://arxiv.org/pdf/2504.01908)  

**Abstract**: Evaluating the quality of synthetic data remains a key challenge for ensuring privacy and utility in data-driven research. In this work, we present an evaluation framework that quantifies how well synthetic data replicates original distributional properties while ensuring privacy. The proposed approach employs a holdout-based benchmarking strategy that facilitates quantitative assessment through low- and high-dimensional distribution comparisons, embedding-based similarity measures, and nearest-neighbor distance metrics. The framework supports various data types and structures, including sequential and contextual information, and enables interpretable quality diagnostics through a set of standardized metrics. These contributions aim to support reproducibility and methodological consistency in benchmarking of synthetic data generation techniques. The code of the framework is available at this https URL. 

---
# Accelerating IoV Intrusion Detection: Benchmarking GPU-Accelerated vs CPU-Based ML Libraries 

**Authors**: Furkan Çolhak, Hasan Coşkun, Tsafac Nkombong Regine Cyrille, Tedi Hoxa, Mert İlhan Ecevit, Mehmet Nafiz Aydın  

**Link**: [PDF](https://arxiv.org/pdf/2504.01905)  

**Abstract**: The Internet of Vehicles (IoV) may face challenging cybersecurity attacks that may require sophisticated intrusion detection systems, necessitating a rapid development and response system. This research investigates the performance advantages of GPU-accelerated libraries (cuML) compared to traditional CPU-based implementations (scikit-learn), focusing on the speed and efficiency required for machine learning models used in IoV threat detection environments. The comprehensive evaluations conducted employ four machine learning approaches (Random Forest, KNN, Logistic Regression, XGBoost) across three distinct IoV security datasets (OTIDS, GIDS, CICIoV2024). Our findings demonstrate that GPU-accelerated implementations dramatically improved computational efficiency, with training times reduced by a factor of up to 159 and prediction speeds accelerated by up to 95 times compared to traditional CPU processing, all while preserving detection accuracy. This remarkable performance breakthrough empowers researchers and security specialists to harness GPU acceleration for creating faster, more effective threat detection systems that meet the urgent real-time security demands of today's connected vehicle networks. 

---
# STAR-1: Safer Alignment of Reasoning LLMs with 1K Data 

**Authors**: Zijun Wang, Haoqin Tu, Yuhan Wang, Juncheng Wu, Jieru Mei, Brian R. Bartoldson, Bhavya Kailkhura, Cihang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2504.01903)  

**Abstract**: This paper introduces STAR-1, a high-quality, just-1k-scale safety dataset specifically designed for large reasoning models (LRMs) like DeepSeek-R1. Built on three core principles -- diversity, deliberative reasoning, and rigorous filtering -- STAR-1 aims to address the critical needs for safety alignment in LRMs. Specifically, we begin by integrating existing open-source safety datasets from diverse sources. Then, we curate safety policies to generate policy-grounded deliberative reasoning samples. Lastly, we apply a GPT-4o-based safety scoring system to select training examples aligned with best practices. Experimental results show that fine-tuning LRMs with STAR-1 leads to an average 40% improvement in safety performance across four benchmarks, while only incurring a marginal decrease (e.g., an average of 1.1%) in reasoning ability measured across five reasoning tasks. Extensive ablation studies further validate the importance of our design principles in constructing STAR-1 and analyze its efficacy across both LRMs and traditional LLMs. Our project page is this https URL. 

---
# Graphically Speaking: Unmasking Abuse in Social Media with Conversation Insights 

**Authors**: Célia Nouri, Jean-Philippe Cointet, Chloé Clavel  

**Link**: [PDF](https://arxiv.org/pdf/2504.01902)  

**Abstract**: Detecting abusive language in social media conversations poses significant challenges, as identifying abusiveness often depends on the conversational context, characterized by the content and topology of preceding comments. Traditional Abusive Language Detection (ALD) models often overlook this context, which can lead to unreliable performance metrics. Recent Natural Language Processing (NLP) methods that integrate conversational context often depend on limited and simplified representations, and report inconsistent results. In this paper, we propose a novel approach that utilize graph neural networks (GNNs) to model social media conversations as graphs, where nodes represent comments, and edges capture reply structures. We systematically investigate various graph representations and context windows to identify the optimal configuration for ALD. Our GNN model outperform both context-agnostic baselines and linear context-aware methods, achieving significant improvements in F1 scores. These findings demonstrate the critical role of structured conversational context and establish GNNs as a robust framework for advancing context-aware abusive language detection. 

---
# Ross3D: Reconstructive Visual Instruction Tuning with 3D-Awareness 

**Authors**: Haochen Wang, Yucheng Zhao, Tiancai Wang, Haoqiang Fan, Xiangyu Zhang, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01901)  

**Abstract**: The rapid development of Large Multimodal Models (LMMs) for 2D images and videos has spurred efforts to adapt these models for interpreting 3D scenes. However, the absence of large-scale 3D vision-language datasets has posed a significant obstacle. To address this issue, typical approaches focus on injecting 3D awareness into 2D LMMs by designing 3D input-level scene representations. This work provides a new perspective. We introduce reconstructive visual instruction tuning with 3D-awareness (Ross3D), which integrates 3D-aware visual supervision into the training procedure. Specifically, it incorporates cross-view and global-view reconstruction. The former requires reconstructing masked views by aggregating overlapping information from other views. The latter aims to aggregate information from all available views to recover Bird's-Eye-View images, contributing to a comprehensive overview of the entire scene. Empirically, Ross3D achieves state-of-the-art performance across various 3D scene understanding benchmarks. More importantly, our semi-supervised experiments demonstrate significant potential in leveraging large amounts of unlabeled 3D vision-only data. 

---
# A novel gesture interaction control method for rehabilitation lower extremity exoskeleton 

**Authors**: Shuang Qiu, Zhongcai Pei, Chen Wang, Jing Zhang, Zhiyong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01888)  

**Abstract**: With the rapid development of Rehabilitation Lower Extremity Robotic Exoskeletons (RLEEX) technology, significant advancements have been made in Human-Robot Interaction (HRI) methods. These include traditional physical HRI methods that are easily recognizable and various bio-electrical signal-based HRI methods that can visualize and predict actions. However, most of these HRI methods are contact-based, facing challenges such as operational complexity, sensitivity to interference, risks associated with implantable devices, and, most importantly, limitations in comfort. These challenges render the interaction less intuitive and natural, which can negatively impact patient motivation for rehabilitation. To address these issues, this paper proposes a novel non-contact gesture interaction control method for RLEEX, based on RGB monocular camera depth estimation. This method integrates three key steps: detecting keypoints, recognizing gestures, and assessing distance, thereby applying gesture information and augmented reality triggering technology to control gait movements of RLEEX. Results indicate that this approach provides a feasible solution to the problems of poor comfort, low reliability, and high latency in HRI for RLEEX platforms. Specifically, it achieves a gesture-controlled exoskeleton motion accuracy of 94.11\% and an average system response time of 0.615 seconds through non-contact HRI. The proposed non-contact HRI method represents a pioneering advancement in control interactions for RLEEX, paving the way for further exploration and development in this field. 

---
# Interpreting Emergent Planning in Model-Free Reinforcement Learning 

**Authors**: Thomas Bush, Stephen Chung, Usman Anwar, Adrià Garriga-Alonso, David Krueger  

**Link**: [PDF](https://arxiv.org/pdf/2504.01871)  

**Abstract**: We present the first mechanistic evidence that model-free reinforcement learning agents can learn to plan. This is achieved by applying a methodology based on concept-based interpretability to a model-free agent in Sokoban -- a commonly used benchmark for studying planning. Specifically, we demonstrate that DRC, a generic model-free agent introduced by Guez et al. (2019), uses learned concept representations to internally formulate plans that both predict the long-term effects of actions on the environment and influence action selection. Our methodology involves: (1) probing for planning-relevant concepts, (2) investigating plan formation within the agent's representations, and (3) verifying that discovered plans (in the agent's representations) have a causal effect on the agent's behavior through interventions. We also show that the emergence of these plans coincides with the emergence of a planning-like property: the ability to benefit from additional test-time compute. Finally, we perform a qualitative analysis of the planning algorithm learned by the agent and discover a strong resemblance to parallelized bidirectional search. Our findings advance understanding of the internal mechanisms underlying planning behavior in agents, which is important given the recent trend of emergent planning and reasoning capabilities in LLMs through RL 

---
# From Code Generation to Software Testing: AI Copilot with Context-Based RAG 

**Authors**: Yuchen Wang, Shangxin Guo, Chee Wei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.01866)  

**Abstract**: The rapid pace of large-scale software development places increasing demands on traditional testing methodologies, often leading to bottlenecks in efficiency, accuracy, and coverage. We propose a novel perspective on software testing by positing bug detection and coding with fewer bugs as two interconnected problems that share a common goal, which is reducing bugs with limited resources. We extend our previous work on AI-assisted programming, which supports code auto-completion and chatbot-powered Q&A, to the realm of software testing. We introduce Copilot for Testing, an automated testing system that synchronizes bug detection with codebase updates, leveraging context-based Retrieval Augmented Generation (RAG) to enhance the capabilities of large language models (LLMs). Our evaluation demonstrates a 31.2% improvement in bug detection accuracy, a 12.6% increase in critical test coverage, and a 10.5% higher user acceptance rate, highlighting the transformative potential of AI-driven technologies in modern software development practices. 

---
# Cross-Lingual Consistency: A Novel Inference Framework for Advancing Reasoning in Large Language Models 

**Authors**: Zhiwei Yu, Tuo Li, Changhong Wang, Hui Chen, Lang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.01857)  

**Abstract**: Chain-of-thought (CoT) has emerged as a critical mechanism for enhancing reasoning capabilities in large language models (LLMs), with self-consistency demonstrating notable promise in boosting performance. However, inherent linguistic biases in multilingual training corpora frequently cause semantic drift and logical inconsistencies, especially in sub-10B parameter LLMs handling complex inference tasks. To overcome these constraints, we propose the Cross-Lingual Consistency (CLC) framework, an innovative inference paradigm that integrates multilingual reasoning paths through majority voting to elevate LLMs' reasoning capabilities. Empirical evaluations on the CMATH dataset reveal CLC's superiority over the conventional self-consistency method, delivering 9.5%, 6.5%, and 6.0% absolute accuracy gains for DeepSeek-Math-7B-Instruct, Qwen2.5-Math-7B-Instruct, and Gemma2-9B-Instruct respectively. Expanding CLC's linguistic scope to 11 diverse languages implies two synergistic benefits: 1) neutralizing linguistic biases in multilingual training corpora through multilingual ensemble voting, 2) escaping monolingual reasoning traps by exploring the broader multilingual solution space. This dual benefits empirically enables more globally optimal reasoning paths compared to monolingual self-consistency baselines, as evidenced by the 4.1%-18.5% accuracy gains using Gemma2-9B-Instruct on the MGSM dataset. 

---
# Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions 

**Authors**: Jinyoung Choi, Junoh Kang, Bohyung Han  

**Link**: [PDF](https://arxiv.org/pdf/2504.01855)  

**Abstract**: Diffusion probabilistic models (DPMs), while effective in generating high-quality samples, often suffer from high computational costs due to their iterative sampling process. To address this, we propose an enhanced ODE-based sampling method for DPMs inspired by Richardson extrapolation, which reduces numerical error and improves convergence rates. Our method, RX-DPM, leverages multiple ODE solutions at intermediate time steps to extrapolate the denoised prediction in DPMs. This significantly enhances the accuracy of estimations for the final sample while maintaining the number of function evaluations (NFEs). Unlike standard Richardson extrapolation, which assumes uniform discretization of the time grid, we develop a more general formulation tailored to arbitrary time step scheduling, guided by local truncation error derived from a baseline sampling method. The simplicity of our approach facilitates accurate estimation of numerical solutions without significant computational overhead, and allows for seamless and convenient integration into various DPMs and solvers. Additionally, RX-DPM provides explicit error estimates, effectively demonstrating the faster convergence as the leading error term's order increases. Through a series of experiments, we show that the proposed method improves the quality of generated samples without requiring additional sampling iterations. 

---
# Code Red! On the Harmfulness of Applying Off-the-shelf Large Language Models to Programming Tasks 

**Authors**: Ali Al-Kaswan, Sebastian Deatc, Begüm Koç, Arie van Deursen, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2504.01850)  

**Abstract**: Nowadays, developers increasingly rely on solutions powered by Large Language Models (LLM) to assist them with their coding tasks. This makes it crucial to align these tools with human values to prevent malicious misuse. In this paper, we propose a comprehensive framework for assessing the potential harmfulness of LLMs within the software engineering domain. We begin by developing a taxonomy of potentially harmful software engineering scenarios and subsequently, create a dataset of prompts based on this taxonomy. To systematically assess the responses, we design and validate an automatic evaluator that classifies the outputs of a variety of LLMs both open-source and closed-source models, as well as general-purpose and code-specific LLMs. Furthermore, we investigate the impact of models size, architecture family, and alignment strategies on their tendency to generate harmful content. The results show significant disparities in the alignment of various LLMs for harmlessness. We find that some models and model families, such as Openhermes, are more harmful than others and that code-specific models do not perform better than their general-purpose counterparts. Notably, some fine-tuned models perform significantly worse than their base-models due to their design choices. On the other side, we find that larger models tend to be more helpful and are less likely to respond with harmful information. These results highlight the importance of targeted alignment strategies tailored to the unique challenges of software engineering tasks and provide a foundation for future work in this critical area. 

---
# YourBench: Easy Custom Evaluation Sets for Everyone 

**Authors**: Sumuk Shashidhar, Clémentine Fourrier, Alina Lozovskia, Thomas Wolf, Gokhan Tur, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2504.01833)  

**Abstract**: Evaluating large language models (LLMs) effectively remains a critical bottleneck, as traditional static benchmarks suffer from saturation and contamination, while human evaluations are costly and slow. This hinders timely or domain-specific assessment, crucial for real-world applications. We introduce YourBench, a novel, open-source framework that addresses these limitations by enabling dynamic, automated generation of reliable, up-to-date, and domain-tailored benchmarks cheaply and without manual annotation, directly from user-provided documents. We demonstrate its efficacy by replicating 7 diverse MMLU subsets using minimal source text, achieving this for under 15 USD in total inference costs while perfectly preserving the relative model performance rankings (Spearman Rho = 1) observed on the original benchmark. To ensure that YourBench generates data grounded in provided input instead of relying on posterior parametric knowledge in models, we also introduce Tempora-0325, a novel dataset of over 7K diverse documents, published exclusively after March 2025. Our comprehensive analysis spans 26 SoTA models from 7 major families across varying scales (3-671B parameters) to validate the quality of generated evaluations through rigorous algorithmic checks (e.g., citation grounding) and human assessments. We release the YourBench library, the Tempora-0325 dataset, 150k+ question answer pairs based on Tempora and all evaluation and inference traces to facilitate reproducible research and empower the community to generate bespoke benchmarks on demand, fostering more relevant and trustworthy LLM evaluation. 

---
# Implicit Bias Injection Attacks against Text-to-Image Diffusion Models 

**Authors**: Huayang Huang, Xiangye Jin, Jiaxu Miao, Yu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.01819)  

**Abstract**: The proliferation of text-to-image diffusion models (T2I DMs) has led to an increased presence of AI-generated images in daily life. However, biased T2I models can generate content with specific tendencies, potentially influencing people's perceptions. Intentional exploitation of these biases risks conveying misleading information to the public. Current research on bias primarily addresses explicit biases with recognizable visual patterns, such as skin color and gender. This paper introduces a novel form of implicit bias that lacks explicit visual features but can manifest in diverse ways across various semantic contexts. This subtle and versatile nature makes this bias challenging to detect, easy to propagate, and adaptable to a wide range of scenarios. We further propose an implicit bias injection attack framework (IBI-Attacks) against T2I diffusion models by precomputing a general bias direction in the prompt embedding space and adaptively adjusting it based on different inputs. Our attack module can be seamlessly integrated into pre-trained diffusion models in a plug-and-play manner without direct manipulation of user input or model retraining. Extensive experiments validate the effectiveness of our scheme in introducing bias through subtle and diverse modifications while preserving the original semantics. The strong concealment and transferability of our attack across various scenarios further underscore the significance of our approach. Code is available at this https URL. 

---
# Rethinking industrial artificial intelligence: a unified foundation framework 

**Authors**: Jay Lee, Hanqi Su  

**Link**: [PDF](https://arxiv.org/pdf/2504.01797)  

**Abstract**: Recent advancement in industrial artificial intelligence (AI) is reshaping the industry, driving smarter manufacturing, predictive maintenance, and intelligent decision-making. However, existing approaches often focus primarily on algorithms and models, overlooking the importance of systematically integrating domain knowledge, data, and models to ensure more comprehensive and effective AI solutions. Therefore, the effective development and deployment of Industrial AI solutions require a more comprehensive and systematic approach. To address this gap, this paper summarizes previous research and rethinks the role of industrial AI and presents a unified industrial AI foundation framework comprising three core modules: knowledge module, data module, and model module. These modules help to extend and enhance the industrial AI methodology platform, supporting various industrial applications. In addition, a case study on rotating machinery diagnosis demonstrates the framework's effectiveness, and several future directions are highlighted for the development of the industrial AI foundation framework. 

---
# CLaP -- State Detection from Time Series 

**Authors**: Arik Ermshaus, Patrick Schäfer, Ulf Leser  

**Link**: [PDF](https://arxiv.org/pdf/2504.01783)  

**Abstract**: The ever-growing amount of sensor data from machines, smart devices, and the environment leads to an abundance of high-resolution, unannotated time series (TS). These recordings encode the recognizable properties of latent states and transitions from physical phenomena that can be modelled as abstract processes. The unsupervised localization and identification of these states and their transitions is the task of time series state detection (TSSD). We introduce CLaP, a new, highly accurate and efficient algorithm for TSSD. It leverages the predictive power of time series classification for TSSD in an unsupervised setting by applying novel self-supervision techniques to detect whether data segments emerge from the same state or not. To this end, CLaP cross-validates a classifier with segment-labelled subsequences to quantify confusion between segments. It merges labels from segments with high confusion, representing the same latent state, if this leads to an increase in overall classification quality. We conducted an experimental evaluation using 391 TS from four benchmarks and found CLaP to be significantly more precise in detecting states than five state-of-the-art competitors. It achieves the best accuracy-runtime tradeoff and is scalable to large TS. We provide a Python implementation of CLaP, which can be deployed in TS analysis workflows. 

---
# Leveraging Embedding Techniques in Multimodal Machine Learning for Mental Illness Assessment 

**Authors**: Abdelrahaman A. Hassan, Abdelrahman A. Ali, Aya E. Fouda, Radwa J. Hanafy, Mohammed E. Fouda  

**Link**: [PDF](https://arxiv.org/pdf/2504.01767)  

**Abstract**: The increasing global prevalence of mental disorders, such as depression and PTSD, requires objective and scalable diagnostic tools. Traditional clinical assessments often face limitations in accessibility, objectivity, and consistency. This paper investigates the potential of multimodal machine learning to address these challenges, leveraging the complementary information available in text, audio, and video data. Our approach involves a comprehensive analysis of various data preprocessing techniques, including novel chunking and utterance-based formatting strategies. We systematically evaluate a range of state-of-the-art embedding models for each modality and employ Convolutional Neural Networks (CNNs) and Bidirectional LSTM Networks (BiLSTMs) for feature extraction. We explore data-level, feature-level, and decision-level fusion techniques, including a novel integration of Large Language Model (LLM) predictions. We also investigate the impact of replacing Multilayer Perceptron classifiers with Support Vector Machines. We extend our analysis to severity prediction using PHQ-8 and PCL-C scores and multi-class classification (considering co-occurring conditions). Our results demonstrate that utterance-based chunking significantly improves performance, particularly for text and audio modalities. Decision-level fusion, incorporating LLM predictions, achieves the highest accuracy, with a balanced accuracy of 94.8% for depression and 96.2% for PTSD detection. The combination of CNN-BiLSTM architectures with utterance-level chunking, coupled with the integration of external LLM, provides a powerful and nuanced approach to the detection and assessment of mental health conditions. Our findings highlight the potential of MMML for developing more accurate, accessible, and personalized mental healthcare tools. 

---
# Dual-stream Transformer-GCN Model with Contextualized Representations Learning for Monocular 3D Human Pose Estimation 

**Authors**: Mingrui Ye, Lianping Yang, Hegui Zhu, Zenghao Zheng, Xin Wang, Yantao Lo  

**Link**: [PDF](https://arxiv.org/pdf/2504.01764)  

**Abstract**: This paper introduces a novel approach to monocular 3D human pose estimation using contextualized representation learning with the Transformer-GCN dual-stream model. Monocular 3D human pose estimation is challenged by depth ambiguity, limited 3D-labeled training data, imbalanced modeling, and restricted model generalization. To address these limitations, our work introduces a groundbreaking motion pre-training method based on contextualized representation learning. Specifically, our method involves masking 2D pose features and utilizing a Transformer-GCN dual-stream model to learn high-dimensional representations through a self-distillation setup. By focusing on contextualized representation learning and spatial-temporal modeling, our approach enhances the model's ability to understand spatial-temporal relationships between postures, resulting in superior generalization. Furthermore, leveraging the Transformer-GCN dual-stream model, our approach effectively balances global and local interactions in video pose estimation. The model adaptively integrates information from both the Transformer and GCN streams, where the GCN stream effectively learns local relationships between adjacent key points and frames, while the Transformer stream captures comprehensive global spatial and temporal features. Our model achieves state-of-the-art performance on two benchmark datasets, with an MPJPE of 38.0mm and P-MPJPE of 31.9mm on Human3.6M, and an MPJPE of 15.9mm on MPI-INF-3DHP. Furthermore, visual experiments on public datasets and in-the-wild videos demonstrate the robustness and generalization capabilities of our approach. 

---
# Style over Substance: Distilled Language Models Reason Via Stylistic Replication 

**Authors**: Philip Lippmann, Jie Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01738)  

**Abstract**: Specialized reasoning language models (RLMs) have demonstrated that scaling test-time computation through detailed reasoning traces significantly enhances performance. Although these traces effectively facilitate knowledge distillation into smaller, instruction-tuned models, the precise nature of transferred reasoning remains unclear. In this study, we investigate to what extent distilled models internalize replicated stylistic patterns during reasoning. To this end, we systematically analyze reasoning traces, identifying structural and lexical patterns that characterize successful reasoning. We then introduce two new datasets -- a dataset of emergent reasoning traces and a synthetic dataset explicitly constructed to replicate these stylistic patterns -- to precisely examine their influence on distilled models' reasoning capabilities. We find that models trained on the synthetic traces achieve comparable performance, indicating that distilled reasoning abilities rely significantly on surface-level patterns. Surprisingly, we observe an increase in performance even when the synthetic traces are altered to lead to the wrong answer. Our findings highlight how stylistic patterns can be leveraged to efficiently enhance LM reasoning across diverse model families. 

---
# AdPO: Enhancing the Adversarial Robustness of Large Vision-Language Models with Preference Optimization 

**Authors**: Chaohu Liu, Tianyi Gui, Yu Liu, Linli Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.01735)  

**Abstract**: Large Vision-Language Models (LVLMs), such as GPT-4o and LLaVA, have recently witnessed remarkable advancements and are increasingly being deployed in real-world applications. However, inheriting the sensitivity of visual neural networks, LVLMs remain vulnerable to adversarial attacks, which can result in erroneous or malicious outputs. While existing efforts utilize adversarial fine-tuning to enhance robustness, they often suffer from performance degradation on clean inputs. In this paper, we proposes AdPO, a novel adversarial defense strategy for LVLMs based on preference optimization. For the first time, we reframe adversarial training as a preference optimization problem, aiming to enhance the model's preference for generating normal outputs on clean inputs while rejecting the potential misleading outputs for adversarial examples. Notably, AdPO achieves this by solely modifying the image encoder, e.g., CLIP ViT, resulting in superior clean and adversarial performance in a variety of downsream tasks. Considering that training involves large language models (LLMs), the computational cost increases significantly. We validate that training on smaller LVLMs and subsequently transferring to larger models can achieve competitive performance while maintaining efficiency comparable to baseline methods. Our comprehensive experiments confirm the effectiveness of the proposed AdPO, which provides a novel perspective for future adversarial defense research. 

---
# DreamActor-M1: Holistic, Expressive and Robust Human Image Animation with Hybrid Guidance 

**Authors**: Yuxuan Luo, Zhengkun Rong, Lizhen Wang, Longhao Zhang, Tianshu Hu, Yongming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.01724)  

**Abstract**: While recent image-based human animation methods achieve realistic body and facial motion synthesis, critical gaps remain in fine-grained holistic controllability, multi-scale adaptability, and long-term temporal coherence, which leads to their lower expressiveness and robustness. We propose a diffusion transformer (DiT) based framework, DreamActor-M1, with hybrid guidance to overcome these limitations. For motion guidance, our hybrid control signals that integrate implicit facial representations, 3D head spheres, and 3D body skeletons achieve robust control of facial expressions and body movements, while producing expressive and identity-preserving animations. For scale adaptation, to handle various body poses and image scales ranging from portraits to full-body views, we employ a progressive training strategy using data with varying resolutions and scales. For appearance guidance, we integrate motion patterns from sequential frames with complementary visual references, ensuring long-term temporal coherence for unseen regions during complex movements. Experiments demonstrate that our method outperforms the state-of-the-art works, delivering expressive results for portraits, upper-body, and full-body generation with robust long-term consistency. Project Page: this https URL. 

---
# InfiniteICL: Breaking the Limit of Context Window Size via Long Short-term Memory Transformation 

**Authors**: Bowen Cao, Deng Cai, Wai Lam  

**Link**: [PDF](https://arxiv.org/pdf/2504.01707)  

**Abstract**: In-context learning (ICL) is critical for large language models (LLMs), but its effectiveness is constrained by finite context windows, particularly in ultra-long contexts. To overcome this, we introduce InfiniteICL, a framework that parallels context and parameters in LLMs with short- and long-term memory in human cognitive systems, focusing on transforming temporary context knowledge into permanent parameter updates. This approach significantly reduces memory usage, maintains robust performance across varying input lengths, and theoretically enables infinite context integration through the principles of context knowledge elicitation, selection, and consolidation. Evaluations demonstrate that our method reduces context length by 90% while achieving 103% average performance of full-context prompting across fact recall, grounded reasoning, and skill acquisition tasks. When conducting sequential multi-turn transformations on complex, real-world contexts (with length up to 2M tokens), our approach surpasses full-context prompting while using only 0.4% of the original contexts. These findings highlight InfiniteICL's potential to enhance the scalability and efficiency of LLMs by breaking the limitations of conventional context window sizes. 

---
# Sky of Unlearning (SoUL): Rewiring Federated Machine Unlearning via Selective Pruning 

**Authors**: Md Mahabub Uz Zaman, Xiang Sun, Jingjing Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.01705)  

**Abstract**: The Internet of Drones (IoD), where drones collaborate in data collection and analysis, has become essential for applications such as surveillance and environmental monitoring. Federated learning (FL) enables drones to train machine learning models in a decentralized manner while preserving data privacy. However, FL in IoD networks is susceptible to attacks like data poisoning and model inversion. Federated unlearning (FU) mitigates these risks by eliminating adversarial data contributions, preventing their influence on the model. This paper proposes sky of unlearning (SoUL), a federated unlearning framework that efficiently removes the influence of unlearned data while maintaining model performance. A selective pruning algorithm is designed to identify and remove neurons influential in unlearning but minimally impact the overall performance of the model. Simulations demonstrate that SoUL outperforms existing unlearning methods, achieves accuracy comparable to full retraining, and reduces computation and communication overhead, making it a scalable and efficient solution for resource-constrained IoD networks. 

---
# Reasoning LLMs for User-Aware Multimodal Conversational Agents 

**Authors**: Hamed Rahimi, Jeanne Cattoni, Meriem Beghili, Mouad Abrini, Mahdi Khoramshahi, Maribel Pino, Mohamed Chetouani  

**Link**: [PDF](https://arxiv.org/pdf/2504.01700)  

**Abstract**: Personalization in social robotics is critical for fostering effective human-robot interactions, yet systems often face the cold start problem, where initial user preferences or characteristics are unavailable. This paper proposes a novel framework called USER-LLM R1 for a user-aware conversational agent that addresses this challenge through dynamic user profiling and model initiation. Our approach integrates chain-of-thought (CoT) reasoning models to iteratively infer user preferences and vision-language models (VLMs) to initialize user profiles from multimodal inputs, enabling personalized interactions from the first encounter. Leveraging a Retrieval-Augmented Generation (RAG) architecture, the system dynamically refines user representations within an inherent CoT process, ensuring contextually relevant and adaptive responses. Evaluations on the ElderlyTech-VQA Bench demonstrate significant improvements in ROUGE-1 (+23.2%), ROUGE-2 (+0.6%), and ROUGE-L (+8%) F1 scores over state-of-the-art baselines, with ablation studies underscoring the impact of reasoning model size on performance. Human evaluations further validate the framework's efficacy, particularly for elderly users, where tailored responses enhance engagement and trust. Ethical considerations, including privacy preservation and bias mitigation, are rigorously discussed and addressed to ensure responsible deployment. 

---
# ToM-RL: Reinforcement Learning Unlocks Theory of Mind in Small LLMs 

**Authors**: Yi-Long Lu, Chunhui Zhang, Jiajun Song, Lifeng Fan, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01698)  

**Abstract**: Recent advancements in rule-based reinforcement learning (RL), applied during the post-training phase of large language models (LLMs), have significantly enhanced their capabilities in structured reasoning tasks such as mathematics and logical inference. However, the effectiveness of RL in social reasoning, particularly in Theory of Mind (ToM), the ability to infer others' mental states, remains largely unexplored. In this study, we demonstrate that RL methods effectively unlock ToM reasoning capabilities even in small-scale LLMs (0.5B to 7B parameters). Using a modest dataset comprising 3200 questions across diverse scenarios, our RL-trained 7B model achieves 84.50\% accuracy on the Hi-ToM benchmark, surpassing models like GPT-4o and DeepSeek-v3 despite significantly fewer parameters. While smaller models ($\leq$3B parameters) suffer from reasoning collapse, larger models (7B parameters) maintain stable performance through consistent belief tracking. Additionally, our RL-based models demonstrate robust generalization to higher-order, out-of-distribution ToM problems, novel textual presentations, and previously unseen datasets. These findings highlight RL's potential to enhance social cognitive reasoning, bridging the gap between structured problem-solving and nuanced social inference in LLMs. 

---
# Segmentation variability and radiomics stability for predicting Triple-Negative Breast Cancer subtype using Magnetic Resonance Imaging 

**Authors**: Isabella Cama, Alejandro Guzmán, Cristina Campi, Michele Piana, Karim Lekadir, Sara Garbarino, Oliver Díaz  

**Link**: [PDF](https://arxiv.org/pdf/2504.01692)  

**Abstract**: Most papers caution against using predictive models for disease stratification based on unselected radiomic features, as these features are affected by contouring variability. Instead, they advocate for the use of the Intraclass Correlation Coefficient (ICC) as a measure of stability for feature selection. However, the direct effect of segmentation variability on the predictive models is rarely studied. This study investigates the impact of segmentation variability on feature stability and predictive performance in radiomics-based prediction of Triple-Negative Breast Cancer (TNBC) subtype using Magnetic Resonance Imaging. A total of 244 images from the Duke dataset were used, with segmentation variability introduced through modifications of manual segmentations. For each mask, explainable radiomic features were selected using the Shapley Additive exPlanations method and used to train logistic regression models. Feature stability across segmentations was assessed via ICC, Pearson's correlation, and reliability scores quantifying the relationship between feature stability and segmentation variability. Results indicate that segmentation accuracy does not significantly impact predictive performance. While incorporating peritumoral information may reduce feature reproducibility, it does not diminish feature predictive capability. Moreover, feature selection in predictive models is not inherently tied to feature stability with respect to segmentation, suggesting that an overreliance on ICC or reliability scores for feature selection might exclude valuable predictive features. 

---
# Token Pruning in Audio Transformers: Optimizing Performance and Decoding Patch Importance 

**Authors**: Taehan Lee, Hyukjun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.01690)  

**Abstract**: Vision Transformers (ViTs) have achieved state-of-the-art performance across various computer vision tasks, but their high computational cost remains a challenge. Token pruning has been proposed to reduce this cost by selectively removing less important tokens. While effective in vision tasks by discarding non-object regions, applying this technique to audio tasks presents unique challenges, as distinguishing relevant from irrelevant regions in time-frequency representations is less straightforward. In this study, for the first time, we applied token pruning to ViT-based audio classification models using Mel-spectrograms and analyzed the trade-offs between model performance and computational cost: TopK token pruning can reduce MAC operations of AudioMAE and AST by 30-40%, with less than a 1% drop in classification accuracy. Our analysis reveals that while high-intensity tokens contribute significantly to model accuracy, low-intensity tokens remain important. In particular, they play a more critical role in general audio classification tasks than in speech-specific tasks. 

---
# K-P Quantum Neural Networks 

**Authors**: Elija Perrier  

**Link**: [PDF](https://arxiv.org/pdf/2504.01673)  

**Abstract**: We present an extension of K-P time-optimal quantum control solutions using global Cartan $KAK$ decompositions for geodesic-based solutions. Extending recent time-optimal \emph{constant-$\theta$} control results, we integrate Cartan methods into equivariant quantum neural network (EQNN) for quantum control tasks. We show that a finite-depth limited EQNN ansatz equipped with Cartan layers can replicate the constant-$\theta$ sub-Riemannian geodesics for K-P problems. We demonstrate how for certain classes of control problem on Riemannian symmetric spaces, gradient-based training using an appropriate cost function converges to certain global time-optimal solutions when satisfying simple regularity conditions. This generalises prior geometric control theory methods and clarifies how optimal geodesic estimation can be performed in quantum machine learning contexts. 

---
# Anomaly Detection for Hybrid Butterfly Subspecies via Probability Filtering 

**Authors**: Bo-Kai Ruan, Yi-Zeng Fang, Hong-Han Shuai, Juinn-Dar Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01671)  

**Abstract**: Detecting butterfly hybrids requires knowledge of the parent subspecies, and the process can be tedious when encountering a new subspecies. This study focuses on a specific scenario where a model trained to recognize hybrid species A can generalize to species B when B biologically mimics A. Since species A and B share similar patterns, we leverage BioCLIP as our feature extractor to capture features based on their taxonomy. Consequently, the algorithm designed for species A can be transferred to B, as their hybrid and non-hybrid patterns exhibit similar relationships. To determine whether a butterfly is a hybrid, we adopt proposed probability filtering and color jittering to augment and simulate the mimicry. With these approaches, we achieve second place in the official development phase. Our code is publicly available at this https URL. 

---
# Market-Oriented Flow Allocation for Thermal Solar Plants: An Auction-Based Methodology with Artificial Intelligence 

**Authors**: Sara Ruiz-Moreno, Antonio J. Gallego, Antonio J. Gallego, Antonio J. Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2504.01652)  

**Abstract**: This paper presents a novel method to optimize thermal balance in parabolic trough collector (PTC) plants. It uses a market-based system to distribute flow among loops combined with an artificial neural network (ANN) to reduce computation and data requirements. This auction-based approach balances loop temperatures, accommodating varying thermal losses and collector efficiencies. Validation across different thermal losses, optical efficiencies, and irradiance conditions-sunny, partially cloudy, and cloudy-show improved thermal power output and intercept factors compared to a no-allocation system. It demonstrates scalability and practicality for large solar thermal plants, enhancing overall performance. The method was first validated through simulations on a realistic solar plant model, then adapted and successfully tested in a 50 MW solar trough plant, demonstrating its advantages. Furthermore, the algorithms have been implemented, commissioned, and are currently operating in 13 commercial solar trough plants. 

---
# Bridge 2D-3D: Uncertainty-aware Hierarchical Registration Network with Domain Alignment 

**Authors**: Zhixin Cheng, Jiacheng Deng, Xinjun Li, Baoqun Yin, Tianzhu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01641)  

**Abstract**: The method for image-to-point cloud registration typically determines the rigid transformation using a coarse-to-fine pipeline. However, directly and uniformly matching image patches with point cloud patches may lead to focusing on incorrect noise patches during matching while ignoring key ones. Moreover, due to the significant differences between image and point cloud modalities, it may be challenging to bridge the domain gap without specific improvements in design. To address the above issues, we innovatively propose the Uncertainty-aware Hierarchical Matching Module (UHMM) and the Adversarial Modal Alignment Module (AMAM). Within the UHMM, we model the uncertainty of critical information in image patches and facilitate multi-level fusion interactions between image and point cloud features. In the AMAM, we design an adversarial approach to reduce the domain gap between image and point cloud. Extensive experiments and ablation studies on RGB-D Scene V2 and 7-Scenes benchmarks demonstrate the superiority of our method, making it a state-of-the-art approach for image-to-point cloud registration tasks. 

---
# Benchmarking the Spatial Robustness of DNNs via Natural and Adversarial Localized Corruptions 

**Authors**: Giulia Marchiori Pietrosanti, Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo  

**Link**: [PDF](https://arxiv.org/pdf/2504.01632)  

**Abstract**: The robustness of DNNs is a crucial factor in safety-critical applications, particularly in complex and dynamic environments where localized corruptions can arise. While previous studies have evaluated the robustness of semantic segmentation (SS) models under whole-image natural or adversarial corruptions, a comprehensive investigation into the spatial robustness of dense vision models under localized corruptions remained underexplored. This paper fills this gap by introducing specialized metrics for benchmarking the spatial robustness of segmentation models, alongside with an evaluation framework to assess the impact of localized corruptions. Furthermore, we uncover the inherent complexity of characterizing worst-case robustness using a single localized adversarial perturbation. To address this, we propose region-aware multi-attack adversarial analysis, a method that enables a deeper understanding of model robustness against adversarial perturbations applied to specific regions. The proposed metrics and analysis were evaluated on 15 segmentation models in driving scenarios, uncovering key insights into the effects of localized corruption in both natural and adversarial forms. The results reveal that models respond to these two types of threats differently; for instance, transformer-based segmentation models demonstrate notable robustness to localized natural corruptions but are highly vulnerable to adversarial ones and vice-versa for CNN-based models. Consequently, we also address the challenge of balancing robustness to both natural and adversarial localized corruptions by means of ensemble models, thereby achieving a broader threat coverage and improved reliability for dense vision tasks. 

---
# Horizon Scans can be accelerated using novel information retrieval and artificial intelligence tools 

**Authors**: Lena Schmidt, Oshin Sharma, Chris Marshall, Sonia Garcia Gonzalez Moral  

**Link**: [PDF](https://arxiv.org/pdf/2504.01627)  

**Abstract**: Introduction: Horizon scanning in healthcare assesses early signals of innovation, crucial for timely adoption. Current horizon scanning faces challenges in efficient information retrieval and analysis, especially from unstructured sources like news, presenting a need for innovative tools. Methodology: The study introduces SCANAR and AIDOC, open-source Python-based tools designed to improve horizon scanning. SCANAR automates the retrieval and processing of news articles, offering functionalities such as de-duplication and unsupervised relevancy ranking. AIDOC aids filtration by leveraging AI to reorder textual data based on relevancy, employing neural networks for semantic similarity, and subsequently prioritizing likely relevant entries for human review. Results: Twelve internal datasets from horizon scans and four external benchmarking datasets were used. SCANAR improved retrieval efficiency by automating processes previously dependent on manual labour. AIDOC displayed work-saving potential, achieving around 62% reduction in manual review efforts at 95% recall. Comparative analysis with benchmarking data showed AIDOC's performance was similar to existing systematic review automation tools, though performance varied depending on dataset characteristics. A smaller case-study on our news datasets shows the potential of ensembling large language models within the active-learning process for faster detection of relevant articles across news datasets. Conclusion: The validation indicates that SCANAR and AIDOC show potential to enhance horizon scanning efficiency by streamlining data retrieval and prioritisation. These tools may alleviate methodological limitations and allow broader, swifter horizon scans. Further studies are suggested to optimize these models and to design new workflows and validation processes that integrate large language models. 

---
# Text Speaks Louder than Vision: ASCII Art Reveals Textual Biases in Vision-Language Models 

**Authors**: Zhaochen Wang, Yujun Cai, Zi Huang, Bryan Hooi, Yiwei Wang, Ming-Hsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01589)  

**Abstract**: Vision-language models (VLMs) have advanced rapidly in processing multimodal information, but their ability to reconcile conflicting signals across modalities remains underexplored. This work investigates how VLMs process ASCII art, a unique medium where textual elements collectively form visual patterns, potentially creating semantic-visual conflicts. We introduce a novel evaluation framework that systematically challenges five state-of-the-art models (including GPT-4o, Claude, and Gemini) using adversarial ASCII art, where character-level semantics deliberately contradict global visual patterns. Our experiments reveal a strong text-priority bias: VLMs consistently prioritize textual information over visual patterns, with visual recognition ability declining dramatically as semantic complexity increases. Various mitigation attempts through visual parameter tuning and prompt engineering yielded only modest improvements, suggesting that this limitation requires architectural-level solutions. These findings uncover fundamental flaws in how current VLMs integrate multimodal information, providing important guidance for future model development while highlighting significant implications for content moderation systems vulnerable to adversarial examples. 

---
# Building Knowledge from Interactions: An LLM-Based Architecture for Adaptive Tutoring and Social Reasoning 

**Authors**: Luca Garello, Giulia Belgiovine, Gabriele Russo, Francesco Rea, Alessandra Sciutti  

**Link**: [PDF](https://arxiv.org/pdf/2504.01588)  

**Abstract**: Integrating robotics into everyday scenarios like tutoring or physical training requires robots capable of adaptive, socially engaging, and goal-oriented interactions. While Large Language Models show promise in human-like communication, their standalone use is hindered by memory constraints and contextual incoherence. This work presents a multimodal, cognitively inspired framework that enhances LLM-based autonomous decision-making in social and task-oriented Human-Robot Interaction. Specifically, we develop an LLM-based agent for a robot trainer, balancing social conversation with task guidance and goal-driven motivation. To further enhance autonomy and personalization, we introduce a memory system for selecting, storing and retrieving experiences, facilitating generalized reasoning based on knowledge built across different interactions. A preliminary HRI user study and offline experiments with a synthetic dataset validate our approach, demonstrating the system's ability to manage complex interactions, autonomously drive training tasks, and build and retrieve contextual memories, advancing socially intelligent robotics. 

---
# Pro-DG: Procedural Diffusion Guidance for Architectural Facade Generation 

**Authors**: Aleksander Plocharski, Jan Swidzinski, Przemyslaw Musialski  

**Link**: [PDF](https://arxiv.org/pdf/2504.01571)  

**Abstract**: We present Pro-DG, a framework for procedurally controllable photo-realistic facade generation that combines a procedural shape grammar with diffusion-based image synthesis. Starting from a single input image, we reconstruct its facade layout using grammar rules, then edit that structure through user-defined transformations. As facades are inherently multi-hierarchical structures, we introduce hierarchical matching procedure that aligns facade structures at different levels which is used to introduce control maps to guide a generative diffusion pipeline. This approach retains local appearance fidelity while accommodating large-scale edits such as floor duplication or window rearrangement. We provide a thorough evaluation, comparing Pro-DG against inpainting-based baselines and synthetic ground truths. Our user study and quantitative measurements indicate improved preservation of architectural identity and higher edit accuracy. Our novel method is the first to integrate neuro-symbolically derived shape-grammars for modeling with modern generative model and highlights the broader potential of such approaches for precise and controllable image manipulation. 

---
# Optimizing Package Delivery with Quantum Annealers: Addressing Time-Windows and Simultaneous Pickup and Delivery 

**Authors**: Eneko Osaba, Esther Villar-Rodriguez, Pablo Miranda-Rodriguez, Antón Asla  

**Link**: [PDF](https://arxiv.org/pdf/2504.01560)  

**Abstract**: Recent research at the intersection of quantum computing and routing problems has been highly prolific. Much of this work focuses on classical problems such as the Traveling Salesman Problem and the Vehicle Routing Problem. The practical applicability of these problems depends on the specific objectives and constraints considered. However, it is undeniable that translating complex real-world requirements into these classical formulations often proves challenging. In this paper, we resort to our previously published quantum-classical technique for addressing real-world-oriented routing problems, known as Quantum for Real Package Delivery (Q4RPD), and elaborate on solving additional realistic problem instances. Accordingly, this paper emphasizes the following characteristics: i) simultaneous pickup and deliveries, ii) time-windows, and iii) mobility restrictions by vehicle type. To illustrate the application of Q4RPD, we have conducted an experimentation comprising seven instances, serving as a demonstration of the newly developed features. 

---
# Hyperbolic Diffusion Recommender Model 

**Authors**: Meng Yuan, Yutian Xiao, Wei Chen, Chu Zhao, Deqing Wang, Fuzhen Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01541)  

**Abstract**: Diffusion models (DMs) have emerged as the new state-of-the-art family of deep generative models. To gain deeper insights into the limitations of diffusion models in recommender systems, we investigate the fundamental structural disparities between images and items. Consequently, items often exhibit distinct anisotropic and directional structures that are less prevalent in images. However, the traditional forward diffusion process continuously adds isotropic Gaussian noise, causing anisotropic signals to degrade into noise, which impairs the semantically meaningful representations in recommender systems.
Inspired by the advancements in hyperbolic spaces, we propose a novel \textit{\textbf{H}yperbolic} \textit{\textbf{D}iffusion} \textit{\textbf{R}ecommender} \textit{\textbf{M}odel} (named HDRM). Unlike existing directional diffusion methods based on Euclidean space, the intrinsic non-Euclidean structure of hyperbolic space makes it particularly well-adapted for handling anisotropic diffusion processes. In particular, we begin by formulating concepts to characterize latent directed diffusion processes within a geometrically grounded hyperbolic space. Subsequently, we propose a novel hyperbolic latent diffusion process specifically tailored for users and items. Drawing upon the natural geometric attributes of hyperbolic spaces, we impose structural restrictions on the space to enhance hyperbolic diffusion propagation, thereby ensuring the preservation of the intrinsic topology of user-item graphs. Extensive experiments on three benchmark datasets demonstrate the effectiveness of HDRM. 

---
# Redefining technology for indigenous languages 

**Authors**: Silvia Fernandez-Sabido, Laura Peniche-Sabido  

**Link**: [PDF](https://arxiv.org/pdf/2504.01522)  

**Abstract**: In this paper, we offer an overview of indigenous languages, identifying the causes of their devaluation and the need for legislation on language rights. We review the technologies used to revitalize these languages, finding that when they come from outside, they often have the opposite effect to what they seek; however, when developed from within communities, they become powerful instruments of expression. We propose that the inclusion of Indigenous knowledge in large language models (LLMs) will enrich the technological landscape, but must be done in a participatory environment that encourages the exchange of knowledge. 

---
# Domain Guidance: A Simple Transfer Approach for a Pre-trained Diffusion Model 

**Authors**: Jincheng Zhong, Xiangcheng Zhang, Jianmin Wang, Mingsheng Long  

**Link**: [PDF](https://arxiv.org/pdf/2504.01521)  

**Abstract**: Recent advancements in diffusion models have revolutionized generative modeling. However, the impressive and vivid outputs they produce often come at the cost of significant model scaling and increased computational demands. Consequently, building personalized diffusion models based on off-the-shelf models has emerged as an appealing alternative. In this paper, we introduce a novel perspective on conditional generation for transferring a pre-trained model. From this viewpoint, we propose *Domain Guidance*, a straightforward transfer approach that leverages pre-trained knowledge to guide the sampling process toward the target domain. Domain Guidance shares a formulation similar to advanced classifier-free guidance, facilitating better domain alignment and higher-quality generations. We provide both empirical and theoretical analyses of the mechanisms behind Domain Guidance. Our experimental results demonstrate its substantial effectiveness across various transfer benchmarks, achieving over a 19.6% improvement in FID and a 23.4% improvement in FD$_\text{DINOv2}$ compared to standard fine-tuning. Notably, existing fine-tuned models can seamlessly integrate Domain Guidance to leverage these benefits, without additional training. 

---
# Training-free Dense-Aligned Diffusion Guidance for Modular Conditional Image Synthesis 

**Authors**: Zixuan Wang, Duo Peng, Feng Chen, Yuwei Yang, Yinjie Lei  

**Link**: [PDF](https://arxiv.org/pdf/2504.01515)  

**Abstract**: Conditional image synthesis is a crucial task with broad applications, such as artistic creation and virtual reality. However, current generative methods are often task-oriented with a narrow scope, handling a restricted condition with constrained applicability. In this paper, we propose a novel approach that treats conditional image synthesis as the modular combination of diverse fundamental condition units. Specifically, we divide conditions into three primary units: text, layout, and drag. To enable effective control over these conditions, we design a dedicated alignment module for each. For the text condition, we introduce a Dense Concept Alignment (DCA) module, which achieves dense visual-text alignment by drawing on diverse textual concepts. For the layout condition, we propose a Dense Geometry Alignment (DGA) module to enforce comprehensive geometric constraints that preserve the spatial configuration. For the drag condition, we introduce a Dense Motion Alignment (DMA) module to apply multi-level motion regularization, ensuring that each pixel follows its desired trajectory without visual artifacts. By flexibly inserting and combining these alignment modules, our framework enhances the model's adaptability to diverse conditional generation tasks and greatly expands its application range. Extensive experiments demonstrate the superior performance of our framework across a variety of conditions, including textual description, segmentation mask (bounding box), drag manipulation, and their combinations. Code is available at this https URL. 

---
# HH-PIM: Dynamic Optimization of Power and Performance with Heterogeneous-Hybrid PIM for Edge AI Devices 

**Authors**: Sangmin Jeon, Kangju Lee, Kyeongwon Lee, Woojoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.01468)  

**Abstract**: Processing-in-Memory (PIM) architectures offer promising solutions for efficiently handling AI applications in energy-constrained edge environments. While traditional PIM designs enhance performance and energy efficiency by reducing data movement between memory and processing units, they are limited in edge devices due to continuous power demands and the storage requirements of large neural network weights in SRAM and DRAM. Hybrid PIM architectures, incorporating non-volatile memories like MRAM and ReRAM, mitigate these limitations but struggle with a mismatch between fixed computing resources and dynamically changing inference workloads. To address these challenges, this study introduces a Heterogeneous-Hybrid PIM (HH-PIM) architecture, comprising high-performance MRAM-SRAM PIM modules and low-power MRAM-SRAM PIM modules. We further propose a data placement optimization algorithm that dynamically allocates data based on computational demand, maximizing energy efficiency. FPGA prototyping and power simulations with processors featuring HH-PIM and other PIM types demonstrate that the proposed HH-PIM achieves up to $60.43$ percent average energy savings over conventional PIMs while meeting application latency requirements. These results confirm the suitability of HH-PIM for adaptive, energy-efficient AI processing in edge devices. 

---
# Probabilistic Curriculum Learning for Goal-Based Reinforcement Learning 

**Authors**: Llewyn Salt, Marcus Gallagher  

**Link**: [PDF](https://arxiv.org/pdf/2504.01459)  

**Abstract**: Reinforcement learning (RL) -- algorithms that teach artificial agents to interact with environments by maximising reward signals -- has achieved significant success in recent years. These successes have been facilitated by advances in algorithms (e.g., deep Q-learning, deep deterministic policy gradients, proximal policy optimisation, trust region policy optimisation, and soft actor-critic) and specialised computational resources such as GPUs and TPUs. One promising research direction involves introducing goals to allow multimodal policies, commonly through hierarchical or curriculum reinforcement learning. These methods systematically decompose complex behaviours into simpler sub-tasks, analogous to how humans progressively learn skills (e.g. we learn to run before we walk, or we learn arithmetic before calculus). However, fully automating goal creation remains an open challenge. We present a novel probabilistic curriculum learning algorithm to suggest goals for reinforcement learning agents in continuous control and navigation tasks. 

---
# BiSeg-SAM: Weakly-Supervised Post-Processing Framework for Boosting Binary Segmentation in Segment Anything Models 

**Authors**: Encheng Su, Hu Cao, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2504.01452)  

**Abstract**: Accurate segmentation of polyps and skin lesions is essential for diagnosing colorectal and skin cancers. While various segmentation methods for polyps and skin lesions using fully supervised deep learning techniques have been developed, the pixel-level annotation of medical images by doctors is both time-consuming and costly. Foundational vision models like the Segment Anything Model (SAM) have demonstrated superior performance; however, directly applying SAM to medical segmentation may not yield satisfactory results due to the lack of domain-specific medical knowledge. In this paper, we propose BiSeg-SAM, a SAM-guided weakly supervised prompting and boundary refinement network for the segmentation of polyps and skin lesions. Specifically, we fine-tune SAM combined with a CNN module to learn local features. We introduce a WeakBox with two functions: automatically generating box prompts for the SAM model and using our proposed Multi-choice Mask-to-Box (MM2B) transformation for rough mask-to-box conversion, addressing the mismatch between coarse labels and precise predictions. Additionally, we apply scale consistency (SC) loss for prediction scale alignment. Our DetailRefine module enhances boundary precision and segmentation accuracy by refining coarse predictions using a limited amount of ground truth labels. This comprehensive approach enables BiSeg-SAM to achieve excellent multi-task segmentation performance. Our method demonstrates significant superiority over state-of-the-art (SOTA) methods when tested on five polyp datasets and one skin cancer dataset. 

---
# PiCo: Jailbreaking Multimodal Large Language Models via $\textbf{Pi}$ctorial $\textbf{Co}$de Contextualization 

**Authors**: Aofan Liu, Lulu Tang, Ting Pan, Yuguo Yin, Bin Wang, Ao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01444)  

**Abstract**: Multimodal Large Language Models (MLLMs), which integrate vision and other modalities into Large Language Models (LLMs), significantly enhance AI capabilities but also introduce new security vulnerabilities. By exploiting the vulnerabilities of the visual modality and the long-tail distribution characteristic of code training data, we present PiCo, a novel jailbreaking framework designed to progressively bypass multi-tiered defense mechanisms in advanced MLLMs. PiCo employs a tier-by-tier jailbreak strategy, using token-level typographic attacks to evade input filtering and embedding harmful intent within programming context instructions to bypass runtime monitoring. To comprehensively assess the impact of attacks, a new evaluation metric is further proposed to assess both the toxicity and helpfulness of model outputs post-attack. By embedding harmful intent within code-style visual instructions, PiCo achieves an average Attack Success Rate (ASR) of 84.13% on Gemini-Pro Vision and 52.66% on GPT-4, surpassing previous methods. Experimental results highlight the critical gaps in current defenses, underscoring the need for more robust strategies to secure advanced MLLMs. 

---
# Refining Interactions: Enhancing Anisotropy in Graph Neural Networks with Language Semantics 

**Authors**: Zhaoxing Li, Xiaoming Zhang, Haifeng Zhang, Chengxiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.01429)  

**Abstract**: The integration of Large Language Models (LLMs) with Graph Neural Networks (GNNs) has recently been explored to enhance the capabilities of Text Attribute Graphs (TAGs). Most existing methods feed textual descriptions of the graph structure or neighbouring nodes' text directly into LLMs. However, these approaches often cause LLMs to treat structural information simply as general contextual text, thus limiting their effectiveness in graph-related tasks. In this paper, we introduce LanSAGNN (Language Semantic Anisotropic Graph Neural Network), a framework that extends the concept of anisotropic GNNs to the natural language level. This model leverages LLMs to extract tailor-made semantic information for node pairs, effectively capturing the unique interactions within node relationships. In addition, we propose an efficient dual-layer LLMs finetuning architecture to better align LLMs' outputs with graph tasks. Experimental results demonstrate that LanSAGNN significantly enhances existing LLM-based methods without increasing complexity while also exhibiting strong robustness against interference. 

---
# MuTri: Multi-view Tri-alignment for OCT to OCTA 3D Image Translation 

**Authors**: Zhuangzhuang Chen, Hualiang Wang, Chubin Ou, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.01428)  

**Abstract**: Optical coherence tomography angiography (OCTA) shows its great importance in imaging microvascular networks by providing accurate 3D imaging of blood vessels, but it relies upon specialized sensors and expensive devices. For this reason, previous works show the potential to translate the readily available 3D Optical Coherence Tomography (OCT) images into 3D OCTA images. However, existing OCTA translation methods directly learn the mapping from the OCT domain to the OCTA domain in continuous and infinite space with guidance from only a single view, i.e., the OCTA project map, resulting in suboptimal results. To this end, we propose the multi-view Tri-alignment framework for OCT to OCTA 3D image translation in discrete and finite space, named MuTri. In the first stage, we pre-train two vector-quantized variational auto-encoder (VQ- VAE) by reconstructing 3D OCT and 3D OCTA data, providing semantic prior for subsequent multi-view guidances. In the second stage, our multi-view tri-alignment facilitates another VQVAE model to learn the mapping from the OCT domain to the OCTA domain in discrete and finite space. Specifically, a contrastive-inspired semantic alignment is proposed to maximize the mutual information with the pre-trained models from OCT and OCTA views, to facilitate codebook learning. Meanwhile, a vessel structure alignment is proposed to minimize the structure discrepancy with the pre-trained models from the OCTA project map view, benefiting from learning the detailed vessel structure information. We also collect the first large-scale dataset, namely, OCTA2024, which contains a pair of OCT and OCTA volumes from 846 subjects. 

---
# FAIRE: Assessing Racial and Gender Bias in AI-Driven Resume Evaluations 

**Authors**: Athena Wen, Tanush Patil, Ansh Saxena, Yicheng Fu, Sean O'Brien, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.01420)  

**Abstract**: In an era where AI-driven hiring is transforming recruitment practices, concerns about fairness and bias have become increasingly important. To explore these issues, we introduce a benchmark, FAIRE (Fairness Assessment In Resume Evaluation), to test for racial and gender bias in large language models (LLMs) used to evaluate resumes across different industries. We use two methods-direct scoring and ranking-to measure how model performance changes when resumes are slightly altered to reflect different racial or gender identities. Our findings reveal that while every model exhibits some degree of bias, the magnitude and direction vary considerably. This benchmark provides a clear way to examine these differences and offers valuable insights into the fairness of AI-based hiring tools. It highlights the urgent need for strategies to reduce bias in AI-driven recruitment. Our benchmark code and dataset are open-sourced at our repository: this https URL. 

---
# TimeSearch: Hierarchical Video Search with Spotlight and Reflection for Human-like Long Video Understanding 

**Authors**: Junwen Pan, Rui Zhang, Xin Wan, Yuan Zhang, Ming Lu, Qi She  

**Link**: [PDF](https://arxiv.org/pdf/2504.01407)  

**Abstract**: Large video-language models (LVLMs) have shown remarkable performance across various video-language tasks. However, they encounter significant challenges when processing long videos because of the large number of video frames involved. Downsampling long videos in either space or time can lead to visual hallucinations, making it difficult to accurately interpret long videos. Motivated by human hierarchical temporal search strategies, we propose \textbf{TimeSearch}, a novel framework enabling LVLMs to understand long videos in a human-like manner. TimeSearch integrates two human-like primitives into a unified autoregressive LVLM: 1) \textbf{Spotlight} efficiently identifies relevant temporal events through a Temporal-Augmented Frame Representation (TAFR), explicitly binding visual features with timestamps; 2) \textbf{Reflection} evaluates the correctness of the identified events, leveraging the inherent temporal self-reflection capabilities of LVLMs. TimeSearch progressively explores key events and prioritizes temporal search based on reflection confidence. Extensive experiments on challenging long-video benchmarks confirm that TimeSearch substantially surpasses previous state-of-the-art, improving the accuracy from 41.8\% to 51.5\% on the LVBench. Additionally, experiments on temporal grounding demonstrate that appropriate TAFR is adequate to effectively stimulate the surprising temporal grounding ability of LVLMs in a simpler yet versatile manner, which improves mIoU on Charades-STA by 11.8\%. The code will be released. 

---
# Generative Retrieval and Alignment Model: A New Paradigm for E-commerce Retrieval 

**Authors**: Ming Pang, Chunyuan Yuan, Xiaoyu He, Zheng Fang, Donghao Xie, Fanyi Qu, Xue Jiang, Changping Peng, Zhangang Lin, Zheng Luo, Jingping Shao  

**Link**: [PDF](https://arxiv.org/pdf/2504.01403)  

**Abstract**: Traditional sparse and dense retrieval methods struggle to leverage general world knowledge and often fail to capture the nuanced features of queries and products. With the advent of large language models (LLMs), industrial search systems have started to employ LLMs to generate identifiers for product retrieval. Commonly used identifiers include (1) static/semantic IDs and (2) product term sets. The first approach requires creating a product ID system from scratch, missing out on the world knowledge embedded within LLMs. While the second approach leverages this general knowledge, the significant difference in word distribution between queries and products means that product-based identifiers often do not align well with user search queries, leading to missed product recalls. Furthermore, when queries contain numerous attributes, these algorithms generate a large number of identifiers, making it difficult to assess their quality, which results in low overall recall efficiency.
To address these challenges, this paper introduces a novel e-commerce retrieval paradigm: the Generative Retrieval and Alignment Model (GRAM). GRAM employs joint training on text information from both queries and products to generate shared text identifier codes, effectively bridging the gap between queries and products. This approach not only enhances the connection between queries and products but also improves inference efficiency. The model uses a co-alignment strategy to generate codes optimized for maximizing retrieval efficiency. Additionally, it introduces a query-product scoring mechanism to compare product values across different codes, further boosting retrieval efficiency. Extensive offline and online A/B testing demonstrates that GRAM significantly outperforms traditional models and the latest generative retrieval models, confirming its effectiveness and practicality. 

---
# ToolACE-R: Tool Learning with Adaptive Self-Refinement 

**Authors**: Xingshan Zeng, Weiwen Liu, Xu Huang, Zezhong Wang, Lingzhi Wang, Liangyou Li, Yasheng Wang, Lifeng Shang, Xin Jiang, Ruiming Tang, Qun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.01400)  

**Abstract**: Tool learning, which allows Large Language Models (LLMs) to leverage external tools for solving complex user tasks, has emerged as a promising avenue for extending model capabilities. However, current approaches primarily focus on data synthesis for fine-tuning LLMs to invoke tools effectively, largely ignoring how to fully stimulate the potential of the model. In this paper, we propose ToolACE-R, a novel method that introduces adaptive self-refinement for tool invocations. Our approach features a model-aware iterative training procedure that progressively incorporates more training samples based on the model's evolving capabilities. Additionally, it allows LLMs to iteratively refine their tool calls, optimizing performance without requiring external feedback. To further enhance computational efficiency, we integrate an adaptive mechanism when scaling the inference time, enabling the model to autonomously determine when to stop the refinement process. We conduct extensive experiments across several benchmark datasets, showing that ToolACE-R achieves competitive performance compared to advanced API-based models, even without any refinement. Furthermore, its performance can be further improved efficiently through adaptive self-refinement. Our results demonstrate the effectiveness of the proposed method, which is compatible with base models of various sizes, offering a promising direction for more efficient tool learning. 

---
# From Easy to Hard: Building a Shortcut for Differentially Private Image Synthesis 

**Authors**: Kecen Li, Chen Gong, Xiaochen Li, Yuzhong Zhao, Xinwen Hou, Tianhao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01395)  

**Abstract**: Differentially private (DP) image synthesis aims to generate synthetic images from a sensitive dataset, alleviating the privacy leakage concerns of organizations sharing and utilizing synthetic images. Although previous methods have significantly progressed, especially in training diffusion models on sensitive images with DP Stochastic Gradient Descent (DP-SGD), they still suffer from unsatisfactory performance. In this work, inspired by curriculum learning, we propose a two-stage DP image synthesis framework, where diffusion models learn to generate DP synthetic images from easy to hard. Unlike existing methods that directly use DP-SGD to train diffusion models, we propose an easy stage in the beginning, where diffusion models learn simple features of the sensitive images. To facilitate this easy stage, we propose to use `central images', simply aggregations of random samples of the sensitive dataset. Intuitively, although those central images do not show details, they demonstrate useful characteristics of all images and only incur minimal privacy costs, thus helping early-phase model training. We conduct experiments to present that on the average of four investigated image datasets, the fidelity and utility metrics of our synthetic images are 33.1% and 2.1% better than the state-of-the-art method. 

---
# Virtual Reality and Artificial Intelligence as Psychological Countermeasures in Space and Other Isolated and Confined Environments: A Scoping Review 

**Authors**: Jennifer Sharp, Joshua Kelson, Daryl South, Anthony Saliba, Muhammad Ashad Kabir  

**Link**: [PDF](https://arxiv.org/pdf/2504.01366)  

**Abstract**: Spaceflight is an isolated and confined environment (ICE) that exposes astronauts to psychological hazards, such as stress, danger, and monotony. Virtual reality (VR) and artificial intelligence (AI) technologies can serve as psychological countermeasures as they can digitally simulate immersive environments, interactive companions, and therapeutic experiences. Our study employs a scoping literature review approach to identify what is currently known about the use and effectiveness of VR and AI-based interventions as psychological countermeasures to improve mood or emotional states in adults in space or other ICEs. Additionally, this review aimed to identify gaps in the knowledge base and whether a systematic review with meta-analysis was warranted. The review included studies where the intervention was used or intended for use in space or other extraterrestrial environments (ICE). Our search strategy yielded 19 studies from 3390 records across seven major databases. All studies focused on VR-based interventions, with no eligible AI-based intervention studies found. VR interventions were found to be effective for relaxation and improving mood, emergency training, as an interactive communication platform, for comparing interior designs, and for enhancing exercise. There were improvements for measures of mood and emotion\n (e.g., anxiety and stress); however, user preferences varied, and some instances of cybersickness were reported. A systematic review with meta-analysis is not recommended due to the heterogeneity of results. There is significant scope for further research into the use of VR for a wider range of mood and emotion variables using standardised assessment instruments. Additionally, the potential application of AI as a psychological countermeasure warrants further investigation. 

---
# Advancing MoE Efficiency: A Collaboration-Constrained Routing (C2R) Strategy for Better Expert Parallelism Design 

**Authors**: Mohan Zhang, Pingzhi Li, Jie Peng, Mufan Qiu, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.01337)  

**Abstract**: Mixture-of-Experts (MoE) has successfully scaled up models while maintaining nearly constant computing costs. By employing a gating network to route input tokens, it selectively activates a subset of expert networks to process the corresponding token embeddings. However, in practice, the efficiency of MoE is challenging to achieve due to two key reasons: imbalanced expert activation, which leads to substantial idle time during model or expert parallelism, and insufficient capacity utilization; massive communication overhead, induced by numerous expert routing combinations in expert parallelism at the system level. Previous works typically formulate it as the load imbalance issue characterized by the gating network favoring certain experts over others or attribute it to static execution which fails to adapt to the dynamic expert workload at runtime. In this paper, we exploit it from a brand new perspective, a higher-order view and analysis of MoE routing policies: expert collaboration and specialization where some experts tend to activate broadly with others (collaborative), while others are more likely to activate only with a specific subset of experts (specialized). Our experiments reveal that most experts tend to be overly collaborative, leading to increased communication overhead from repeatedly sending tokens to different accelerators. To this end, we propose a novel collaboration-constrained routing (C2R) strategy to encourage more specialized expert groups, as well as to improve expert utilization, and present an efficient implementation of MoE that further leverages expert specialization. We achieve an average performance improvement of 0.51% and 0.33% on LLaMA-MoE and Qwen-MoE respectively across ten downstream NLP benchmarks, and reduce the all2all communication costs between GPUs, bringing an extra 20%-30% total running time savings on top of the existing SoTA, i.e. MegaBlocks. 

---
# CFMD: Dynamic Cross-layer Feature Fusion for Salient Object Detection 

**Authors**: Jin Lian, Zhongyu Wan, Ming Gao, JunFeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.01326)  

**Abstract**: Cross-layer feature pyramid networks (CFPNs) have achieved notable progress in multi-scale feature fusion and boundary detail preservation for salient object detection. However, traditional CFPNs still suffer from two core limitations: (1) a computational bottleneck caused by complex feature weighting operations, and (2) degraded boundary accuracy due to feature blurring in the upsampling process. To address these challenges, we propose CFMD, a novel cross-layer feature pyramid network that introduces two key innovations. First, we design a context-aware feature aggregation module (CFLMA), which incorporates the state-of-the-art Mamba architecture to construct a dynamic weight distribution mechanism. This module adaptively adjusts feature importance based on image context, significantly improving both representation efficiency and generalization. Second, we introduce an adaptive dynamic upsampling unit (CFLMD) that preserves spatial details during resolution recovery. By adjusting the upsampling range dynamically and initializing with a bilinear strategy, the module effectively reduces feature overlap and maintains fine-grained boundary structures. Extensive experiments on three standard benchmarks using three mainstream backbone networks demonstrate that CFMD achieves substantial improvements in pixel-level accuracy and boundary segmentation quality, especially in complex scenes. The results validate the effectiveness of CFMD in jointly enhancing computational efficiency and segmentation performance, highlighting its strong potential in salient object detection tasks. 

---
# On Data Synthesis and Post-training for Visual Abstract Reasoning 

**Authors**: Ke Zhu, Yu Wang, Jiangjiang Liu, Qunyi Xie, Shanshan Liu, Gang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01324)  

**Abstract**: This paper is a pioneering work attempting to address abstract visual reasoning (AVR) problems for large vision-language models (VLMs). We make a common LLaVA-NeXT 7B model capable of perceiving and reasoning about specific AVR problems, surpassing both open-sourced (e.g., Qwen-2-VL-72B) and closed-sourced powerful VLMs (e.g., GPT-4o) with significant margin. This is a great breakthrough since almost all previous VLMs fail or show nearly random performance on representative AVR benchmarks. Our key success is our innovative data synthesis and post-training process, aiming to fully relieve the task difficulty and elicit the model to learn, step by step. Our 7B model is also shown to be behave well on AVR without sacrificing common multimodal comprehension abilities. We hope our paper could serve as an early effort in this area and would inspire further research in abstract visual reasoning. 

---
# COST: Contrastive One-Stage Transformer for Vision-Language Small Object Tracking 

**Authors**: Chunhui Zhang, Li Liu, Jialin Gao, Xin Sun, Hao Wen, Xi Zhou, Shiming Ge, Yanfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01321)  

**Abstract**: Transformer has recently demonstrated great potential in improving vision-language (VL) tracking algorithms. However, most of the existing VL trackers rely on carefully designed mechanisms to perform the multi-stage multi-modal fusion. Additionally, direct multi-modal fusion without alignment ignores distribution discrepancy between modalities in feature space, potentially leading to suboptimal representations. In this work, we propose COST, a contrastive one-stage transformer fusion framework for VL tracking, aiming to learn semantically consistent and unified VL representations. Specifically, we introduce a contrastive alignment strategy that maximizes mutual information (MI) between a video and its corresponding language description. This enables effective cross-modal alignment, yielding semantically consistent features in the representation space. By leveraging a visual-linguistic transformer, we establish an efficient multi-modal fusion and reasoning mechanism, empirically demonstrating that a simple stack of transformer encoders effectively enables unified VL representations. Moreover, we contribute a newly collected VL tracking benchmark dataset for small object tracking, named VL-SOT500, with bounding boxes and language descriptions. Our dataset comprises two challenging subsets, VL-SOT230 and VL-SOT270, dedicated to evaluating generic and high-speed small object tracking, respectively. Small object tracking is notoriously challenging due to weak appearance and limited features, and this dataset is, to the best of our knowledge, the first to explore the usage of language cues to enhance visual representation for small object tracking. Extensive experiments demonstrate that COST achieves state-of-the-art performance on five existing VL tracking datasets, as well as on our proposed VL-SOT500 dataset. Source codes and dataset will be made publicly available. 

---
# Adaptive Rectification Sampling for Test-Time Compute Scaling 

**Authors**: Zhendong Tan, Xingjun Zhang, Chaoyi Hu, Yancheng Pan, Shaoxun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01317)  

**Abstract**: The newly released OpenAI-o1 and DeepSeek-R1 have demonstrated that test-time scaling can significantly improve model performance, especially in complex tasks such as logical reasoning. Common test-time scaling methods involve generating more chain of thoughts (CoTs) or longer CoTs with self-correction. However, while self-correction can improve performance, it may lead to significant token waste and reduce readability of the CoT if the reasoning steps are already correct. To demonstrate that large language models (LLMs) can rectify errors at a more fine-grained level, we propose Adaptive Rectification Sampling (AR-Sampling), which can guide the LLMs to self-correction at the appropriate step. AR-Sampling leverages a process-supervised reward model (PRM) as a verifier and constructed trigger sentences to guide the model in adaptive step-level rethinking. Through the experiments on GSM8K and MATH500, it indicate that our approach enables the models to rethink in more fine-grained level, improving the accuracy of solutions, while generating a reasonable number of additional tokens. 

---
# Biomedical Question Answering via Multi-Level Summarization on a Local Knowledge Graph 

**Authors**: Lingxiao Guan, Yuanhao Huang, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.01309)  

**Abstract**: In Question Answering (QA), Retrieval Augmented Generation (RAG) has revolutionized performance in various domains. However, how to effectively capture multi-document relationships, particularly critical for biomedical tasks, remains an open question. In this work, we propose a novel method that utilizes propositional claims to construct a local knowledge graph from retrieved documents. Summaries are then derived via layerwise summarization from the knowledge graph to contextualize a small language model to perform QA. We achieved comparable or superior performance with our method over RAG baselines on several biomedical QA benchmarks. We also evaluated each individual step of our methodology over a targeted set of metrics, demonstrating its effectiveness. 

---
# Scaling Test-Time Inference with Policy-Optimized, Dynamic Retrieval-Augmented Generation via KV Caching and Decoding 

**Authors**: Sakhinana Sagar Srinivas, Venkataramana Runkana  

**Link**: [PDF](https://arxiv.org/pdf/2504.01281)  

**Abstract**: We present a comprehensive framework for enhancing Retrieval-Augmented Generation (RAG) systems through dynamic retrieval strategies and reinforcement fine-tuning. This approach significantly improves large language models on knowledge-intensive tasks, including opendomain question answering and complex reasoning. Our framework integrates two complementary techniques: Policy-Optimized RetrievalAugmented Generation (PORAG), which optimizes the use of retrieved information, and Adaptive Token-Layer Attention Scoring (ATLAS), which dynamically determines retrieval timing and content based on contextual needs. Together, these techniques enhance both the utilization and relevance of retrieved content, improving factual accuracy and response quality. Designed as a lightweight solution compatible with any Transformer-based LLM without requiring additional training, our framework excels in knowledge-intensive tasks, boosting output accuracy in RAG settings. We further propose CRITIC, a novel method to selectively compress key-value caches by token importance, mitigating memory bottlenecks in long-context applications. The framework also incorporates test-time scaling techniques to dynamically balance reasoning depth and computational resources, alongside optimized decoding strategies for faster inference. Experiments on benchmark datasets show that our framework reduces hallucinations, strengthens domain-specific reasoning, and achieves significant efficiency and scalability gains over traditional RAG systems. This integrated approach advances the development of robust, efficient, and scalable RAG systems across diverse applications. 

---
# Automated Factual Benchmarking for In-Car Conversational Systems using Large Language Models 

**Authors**: Rafael Giebisch, Ken E. Friedl, Lev Sorokin, Andrea Stocco  

**Link**: [PDF](https://arxiv.org/pdf/2504.01248)  

**Abstract**: In-car conversational systems bring the promise to improve the in-vehicle user experience. Modern conversational systems are based on Large Language Models (LLMs), which makes them prone to errors such as hallucinations, i.e., inaccurate, fictitious, and therefore factually incorrect information. In this paper, we present an LLM-based methodology for the automatic factual benchmarking of in-car conversational systems. We instantiate our methodology with five LLM-based methods, leveraging ensembling techniques and diverse personae to enhance agreement and minimize hallucinations. We use our methodology to evaluate CarExpert, an in-car retrieval-augmented conversational question answering system, with respect to the factual correctness to a vehicle's manual. We produced a novel dataset specifically created for the in-car domain, and tested our methodology against an expert evaluation. Our results show that the combination of GPT-4 with the Input Output Prompting achieves over 90 per cent factual correctness agreement rate with expert evaluations, other than being the most efficient approach yielding an average response time of 4.5s. Our findings suggest that LLM-based testing constitutes a viable approach for the validation of conversational systems regarding their factual correctness. 

---
# Dynamic Graph Structure Estimation for Learning Multivariate Point Process using Spiking Neural Networks 

**Authors**: Biswadeep Chakraborty, Hemant Kumawat, Beomseok Kang, Saibal Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2504.01246)  

**Abstract**: Modeling and predicting temporal point processes (TPPs) is critical in domains such as neuroscience, epidemiology, finance, and social sciences. We introduce the Spiking Dynamic Graph Network (SDGN), a novel framework that leverages the temporal processing capabilities of spiking neural networks (SNNs) and spike-timing-dependent plasticity (STDP) to dynamically estimate underlying spatio-temporal functional graphs. Unlike existing methods that rely on predefined or static graph structures, SDGN adapts to any dataset by learning dynamic spatio-temporal dependencies directly from the event data, enhancing generalizability and robustness. While SDGN offers significant improvements over prior methods, we acknowledge its limitations in handling dense graphs and certain non-Gaussian dependencies, providing opportunities for future refinement. Our evaluations, conducted on both synthetic and real-world datasets including NYC Taxi, 911, Reddit, and Stack Overflow, demonstrate that SDGN achieves superior predictive accuracy while maintaining computational efficiency. Furthermore, we include ablation studies to highlight the contributions of its core components. 

---
# FUSION: Frequency-guided Underwater Spatial Image recOnstructioN 

**Authors**: Jaskaran Singh Walia, Shravan Venkatraman, Pavithra LK  

**Link**: [PDF](https://arxiv.org/pdf/2504.01243)  

**Abstract**: Underwater images suffer from severe degradations, including color distortions, reduced visibility, and loss of structural details due to wavelength-dependent attenuation and scattering. Existing enhancement methods primarily focus on spatial-domain processing, neglecting the frequency domain's potential to capture global color distributions and long-range dependencies. To address these limitations, we propose FUSION, a dual-domain deep learning framework that jointly leverages spatial and frequency domain information. FUSION independently processes each RGB channel through multi-scale convolutional kernels and adaptive attention mechanisms in the spatial domain, while simultaneously extracting global structural information via FFT-based frequency attention. A Frequency Guided Fusion module integrates complementary features from both domains, followed by inter-channel fusion and adaptive channel recalibration to ensure balanced color distributions. Extensive experiments on benchmark datasets (UIEB, EUVP, SUIM-E) demonstrate that FUSION achieves state-of-the-art performance, consistently outperforming existing methods in reconstruction fidelity (highest PSNR of 23.717 dB and SSIM of 0.883 on UIEB), perceptual quality (lowest LPIPS of 0.112 on UIEB), and visual enhancement metrics (best UIQM of 3.414 on UIEB), while requiring significantly fewer parameters (0.28M) and lower computational complexity, demonstrating its suitability for real-time underwater imaging applications. 

---
# TenAd: A Tensor-based Low-rank Black Box Adversarial Attack for Video Classification 

**Authors**: Kimia haghjooei, Mansoor Rezghi  

**Link**: [PDF](https://arxiv.org/pdf/2504.01228)  

**Abstract**: Deep learning models have achieved remarkable success in computer vision but remain vulnerable to adversarial attacks, particularly in black-box settings where model details are unknown. Existing adversarial attack methods(even those works with key frames) often treat video data as simple vectors, ignoring their inherent multi-dimensional structure, and require a large number of queries, making them inefficient and detectable. In this paper, we propose \textbf{TenAd}, a novel tensor-based low-rank adversarial attack that leverages the multi-dimensional properties of video data by representing videos as fourth-order tensors. By exploiting low-rank attack, our method significantly reduces the search space and the number of queries needed to generate adversarial examples in black-box settings. Experimental results on standard video classification datasets demonstrate that \textbf{TenAd} effectively generates imperceptible adversarial perturbations while achieving higher attack success rates and query efficiency compared to state-of-the-art methods. Our approach outperforms existing black-box adversarial attacks in terms of success rate, query efficiency, and perturbation imperceptibility, highlighting the potential of tensor-based methods for adversarial attacks on video models. 

---
# A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates 

**Authors**: Gonçalo Gomes, Chrysoula Zerva, Bruno Martins  

**Link**: [PDF](https://arxiv.org/pdf/2504.01225)  

**Abstract**: This study explores current limitations of learned image captioning evaluation metrics, specifically the lack of granular assessment for individual word misalignments within captions, and the reliance on single-point quality estimates without considering uncertainty. To address these limitations, we propose a simple yet effective strategy for generating and calibrating CLIPScore distributions. Leveraging a model-agnostic conformal risk control framework, we calibrate CLIPScore values for task-specific control variables, to tackle the aforementioned two limitations. Experimental results demonstrate that using conformal risk control, over the distributions produced with simple methods such as input masking, can achieve competitive performance compared to more complex approaches. Our method effectively detects misaligned words, while providing formal guarantees aligned with desired risk levels, and improving the correlation between uncertainty estimations and prediction errors, thus enhancing the overall reliability of caption evaluation metrics. 

---
# Detecting PTSD in Clinical Interviews: A Comparative Analysis of NLP Methods and Large Language Models 

**Authors**: Feng Chen, Dror Ben-Zeev, Gillian Sparks, Arya Kadakia, Trevor Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2504.01216)  

**Abstract**: Post-Traumatic Stress Disorder (PTSD) remains underdiagnosed in clinical settings, presenting opportunities for automated detection to identify patients. This study evaluates natural language processing approaches for detecting PTSD from clinical interview transcripts. We compared general and mental health-specific transformer models (BERT/RoBERTa), embedding-based methods (SentenceBERT/LLaMA), and large language model prompting strategies (zero-shot/few-shot/chain-of-thought) using the DAIC-WOZ dataset. Domain-specific models significantly outperformed general models (Mental-RoBERTa F1=0.643 vs. RoBERTa-base 0.485). LLaMA embeddings with neural networks achieved the highest performance (F1=0.700). Zero-shot prompting using DSM-5 criteria yielded competitive results without training data (F1=0.657). Performance varied significantly across symptom severity and comorbidity status, with higher accuracy for severe PTSD cases and patients with comorbid depression. Our findings highlight the potential of domain-adapted embeddings and LLMs for scalable screening while underscoring the need for improved detection of nuanced presentations and offering insights for developing clinically viable AI tools for PTSD assessment. 

---
# PolygoNet: Leveraging Simplified Polygonal Representation for Effective Image Classification 

**Authors**: Salim Khazem, Jeremy Fix, Cédric Pradalier  

**Link**: [PDF](https://arxiv.org/pdf/2504.01214)  

**Abstract**: Deep learning models have achieved significant success in various image related tasks. However, they often encounter challenges related to computational complexity and overfitting. In this paper, we propose an efficient approach that leverages polygonal representations of images using dominant points or contour coordinates. By transforming input images into these compact forms, our method significantly reduces computational requirements, accelerates training, and conserves resources making it suitable for real time and resource constrained applications. These representations inherently capture essential image features while filtering noise, providing a natural regularization effect that mitigates overfitting. The resulting lightweight models achieve performance comparable to state of the art methods using full resolution images while enabling deployment on edge devices. Extensive experiments on benchmark datasets validate the effectiveness of our approach in reducing complexity, improving generalization, and facilitating edge computing applications. This work demonstrates the potential of polygonal representations in advancing efficient and scalable deep learning solutions for real world scenarios. The code for the experiments of the paper is provided in this https URL. 

---
# Lightweight Deep Models for Dermatological Disease Detection: A Study on Instance Selection and Channel Optimization 

**Authors**: Ian Mateos Gonzalez, Estefani Jaramilla Nava, Abraham Sánchez Morales, Jesús García-Ramírez, Ricardo Ramos-Aguilar  

**Link**: [PDF](https://arxiv.org/pdf/2504.01208)  

**Abstract**: The identification of dermatological disease is an important problem in Mexico according with different studies. Several works in literature use the datasets of different repositories without applying a study of the data behavior, especially in medical images domain. In this work, we propose a methodology to preprocess dermaMNIST dataset in order to improve its quality for the classification stage, where we use lightweight convolutional neural networks. In our results, we reduce the number of instances for the neural network training obtaining a similar performance of models as ResNet. 

---
# Epistemic Alignment: A Mediating Framework for User-LLM Knowledge Delivery 

**Authors**: Nicholas Clark, Hua Shen, Bill Howe, Tanushree Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2504.01205)  

**Abstract**: LLMs increasingly serve as tools for knowledge acquisition, yet users cannot effectively specify how they want information presented. When users request that LLMs "cite reputable sources," "express appropriate uncertainty," or "include multiple perspectives," they discover that current interfaces provide no structured way to articulate these preferences. The result is prompt sharing folklore: community-specific copied prompts passed through trust relationships rather than based on measured efficacy. We propose the Epistemic Alignment Framework, a set of ten challenges in knowledge transmission derived from the philosophical literature of epistemology, concerning issues such as evidence quality assessment and calibration of testimonial reliance. The framework serves as a structured intermediary between user needs and system capabilities, creating a common vocabulary to bridge the gap between what users want and what systems deliver. Through a thematic analysis of custom prompts and personalization strategies shared on online communities where these issues are actively discussed, we find users develop elaborate workarounds to address each of the challenges. We then apply our framework to two prominent model providers, OpenAI and Anthropic, through content analysis of their documented policies and product features. Our analysis shows that while these providers have partially addressed the challenges we identified, they fail to establish adequate mechanisms for specifying epistemic preferences, lack transparency about how preferences are implemented, and offer no verification tools to confirm whether preferences were followed. For AI developers, the Epistemic Alignment Framework offers concrete guidance for supporting diverse approaches to knowledge; for users, it works toward information delivery that aligns with their specific needs rather than defaulting to one-size-fits-all approaches. 

---
# Medical large language models are easily distracted 

**Authors**: Krithik Vishwanath, Anton Alyakin, Daniel Alexander Alber, Jin Vivian Lee, Douglas Kondziolka, Eric Karl Oermann  

**Link**: [PDF](https://arxiv.org/pdf/2504.01201)  

**Abstract**: Large language models (LLMs) have the potential to transform medicine, but real-world clinical scenarios contain extraneous information that can hinder performance. The rise of assistive technologies like ambient dictation, which automatically generates draft notes from live patient encounters, has the potential to introduce additional noise making it crucial to assess the ability of LLM's to filter relevant data. To investigate this, we developed MedDistractQA, a benchmark using USMLE-style questions embedded with simulated real-world distractions. Our findings show that distracting statements (polysemous words with clinical meanings used in a non-clinical context or references to unrelated health conditions) can reduce LLM accuracy by up to 17.9%. Commonly proposed solutions to improve model performance such as retrieval-augmented generation (RAG) and medical fine-tuning did not change this effect and in some cases introduced their own confounders and further degraded performance. Our findings suggest that LLMs natively lack the logical mechanisms necessary to distinguish relevant from irrelevant clinical information, posing challenges for real-world applications. MedDistractQA and our results highlights the need for robust mitigation strategies to enhance LLM resilience to extraneous information. 

---
# $μ$KE: Matryoshka Unstructured Knowledge Editing of Large Language Models 

**Authors**: Zian Su, Ziyang Huang, Kaiyuan Zhang, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01196)  

**Abstract**: Large language models (LLMs) have emerged as powerful knowledge bases yet are limited by static training data, leading to issues such as hallucinations and safety risks. Editing a model's internal knowledge through the locate-and-edit paradigm has proven a cost-effective alternative to retraining, though current unstructured approaches, especially window-based autoregressive methods, often disrupt the causal dependency between early memory updates and later output tokens. In this work, we first theoretically analyze these limitations and then introduce Matryoshka Unstructured Knowledge Editing ($\mu$KE), a novel memory update mechanism that preserves such dependencies via a Matryoshka-style objective and adaptive loss coefficients. Empirical evaluations on two models across four benchmarks demonstrate that $\mu$KE improves edit efficacy by up to 12.33% over state-of-the-art methods, and remain robust when applied to diverse formatted edits, underscoring its potential for effective unstructured knowledge editing in LLMs. 

---
# Neural Approaches to SAT Solving: Design Choices and Interpretability 

**Authors**: David Mojžíšek, Jan Hůla, Ziwei Li, Ziyu Zhou, Mikoláš Janota  

**Link**: [PDF](https://arxiv.org/pdf/2504.01173)  

**Abstract**: In this contribution, we provide a comprehensive evaluation of graph neural networks applied to Boolean satisfiability problems, accompanied by an intuitive explanation of the mechanisms enabling the model to generalize to different instances. We introduce several training improvements, particularly a novel closest assignment supervision method that dynamically adapts to the model's current state, significantly enhancing performance on problems with larger solution spaces. Our experiments demonstrate the suitability of variable-clause graph representations with recurrent neural network updates, which achieve good accuracy on SAT assignment prediction while reducing computational demands. We extend the base graph neural network into a diffusion model that facilitates incremental sampling and can be effectively combined with classical techniques like unit propagation. Through analysis of embedding space patterns and optimization trajectories, we show how these networks implicitly perform a process very similar to continuous relaxations of MaxSAT, offering an interpretable view of their reasoning process. This understanding guides our design choices and explains the ability of recurrent architectures to scale effectively at inference time beyond their training distribution, which we demonstrate with test-time scaling experiments. 

---
# Catch Me if You Search: When Contextual Web Search Results Affect the Detection of Hallucinations 

**Authors**: Mahjabin Nahar, Eun-Ju Lee, Jin Won Park, Dongwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.01153)  

**Abstract**: While we increasingly rely on large language models (LLMs) for various tasks, these models are known to produce inaccurate content or 'hallucinations' with potentially disastrous consequences. The recent integration of web search results into LLMs prompts the question of whether people utilize them to verify the generated content, thereby avoiding falling victim to hallucinations. This study (N = 560) investigated how the provision of search results, either static (fixed search results) or dynamic (participant-driven searches), affect participants' perceived accuracy and confidence in evaluating LLM-generated content (i.e., genuine, minor hallucination, major hallucination), compared to the control condition (no search results). Findings indicate that participants in both static and dynamic conditions (vs. control) rated hallucinated content to be less accurate. However, those in the dynamic condition rated genuine content as more accurate and demonstrated greater overall confidence in their assessments than those in the static or control conditions. In addition, those higher in need for cognition (NFC) rated major hallucinations to be less accurate than low NFC participants, with no corresponding difference for genuine content or minor hallucinations. These results underscore the potential benefits of integrating web search results into LLMs for the detection of hallucinations, as well as the need for a more nuanced approach when developing human-centered systems, taking user characteristics into account. 

---
# Is the Top Still Spinning? Evaluating Subjectivity in Narrative Understanding 

**Authors**: Melanie Subbiah, Akankshya Mishra, Grace Kim, Liyan Tang, Greg Durrett, Kathleen McKeown  

**Link**: [PDF](https://arxiv.org/pdf/2504.01132)  

**Abstract**: Determining faithfulness of a claim to a source document is an important problem across many domains. This task is generally treated as a binary judgment of whether the claim is supported or unsupported in relation to the source. In many cases, though, whether a claim is supported can be ambiguous. For instance, it may depend on making inferences from given evidence, and different people can reasonably interpret the claim as either supported or unsupported based on their agreement with those inferences. Forcing binary labels upon such claims lowers the reliability of evaluation. In this work, we reframe the task to manage the subjectivity involved with factuality judgments of ambiguous claims. We introduce LLM-generated edits of summaries as a method of providing a nuanced evaluation of claims: how much does a summary need to be edited to be unambiguous? Whether a claim gets rewritten and how much it changes can be used as an automatic evaluation metric, the Ambiguity Rewrite Metric (ARM), with a much richer feedback signal than a binary judgment of faithfulness. We focus on the area of narrative summarization as it is particularly rife with ambiguity and subjective interpretation. We show that ARM produces a 21% absolute improvement in annotator agreement on claim faithfulness, indicating that subjectivity is reduced. 

---
# RipVIS: Rip Currents Video Instance Segmentation Benchmark for Beach Monitoring and Safety 

**Authors**: Andrei Dumitriu, Florin Tatui, Florin Miron, Aakash Ralhan, Radu Tudor Ionescu, Radu Timofte  

**Link**: [PDF](https://arxiv.org/pdf/2504.01128)  

**Abstract**: Rip currents are strong, localized and narrow currents of water that flow outwards into the sea, causing numerous beach-related injuries and fatalities worldwide. Accurate identification of rip currents remains challenging due to their amorphous nature and the lack of annotated data, which often requires expert knowledge. To address these issues, we present RipVIS, a large-scale video instance segmentation benchmark explicitly designed for rip current segmentation. RipVIS is an order of magnitude larger than previous datasets, featuring $184$ videos ($212,328$ frames), of which $150$ videos ($163,528$ frames) are with rip currents, collected from various sources, including drones, mobile phones, and fixed beach cameras. Our dataset encompasses diverse visual contexts, such as wave-breaking patterns, sediment flows, and water color variations, across multiple global locations, including USA, Mexico, Costa Rica, Portugal, Italy, Greece, Romania, Sri Lanka, Australia and New Zealand. Most videos are annotated at $5$ FPS to ensure accuracy in dynamic scenarios, supplemented by an additional $34$ videos ($48,800$ frames) without rip currents. We conduct comprehensive experiments with Mask R-CNN, Cascade Mask R-CNN, SparseInst and YOLO11, fine-tuning these models for the task of rip current segmentation. Results are reported in terms of multiple metrics, with a particular focus on the $F_2$ score to prioritize recall and reduce false negatives. To enhance segmentation performance, we introduce a novel post-processing step based on Temporal Confidence Aggregation (TCA). RipVIS aims to set a new standard for rip current segmentation, contributing towards safer beach environments. We offer a benchmark website to share data, models, and results with the research community, encouraging ongoing collaboration and future contributions, at this https URL. 

---
# ffstruc2vec: Flat, Flexible and Scalable Learning of Node Representations from Structural Identities 

**Authors**: Mario Heidrich, Jeffrey Heidemann, Rüdiger Buchkremer, Gonzalo Wandosell Fernández de Bobadilla  

**Link**: [PDF](https://arxiv.org/pdf/2504.01122)  

**Abstract**: Node embedding refers to techniques that generate low-dimensional vector representations of nodes in a graph while preserving specific properties of the nodes. A key challenge in the field is developing scalable methods that can preserve structural properties suitable for the required types of structural patterns of a given downstream application task. While most existing methods focus on preserving node proximity, those that do preserve structural properties often lack the flexibility to preserve various types of structural patterns required by downstream application tasks. This paper introduces ffstruc2vec, a scalable deep-learning framework for learning node embedding vectors that preserve structural identities. Its flat, efficient architecture allows high flexibility in capturing diverse types of structural patterns, enabling broad adaptability to various downstream application tasks. The proposed framework significantly outperforms existing approaches across diverse unsupervised and supervised tasks in practical applications. Moreover, ffstruc2vec enables explainability by quantifying how individual structural patterns influence task outcomes, providing actionable interpretation. To our knowledge, no existing framework combines this level of flexibility, scalability, and structural interpretability, underscoring its unique capabilities. 

---
# Multilingual and Multi-Accent Jailbreaking of Audio LLMs 

**Authors**: Jaechul Roh, Virat Shejwalkar, Amir Houmansadr  

**Link**: [PDF](https://arxiv.org/pdf/2504.01094)  

**Abstract**: Large Audio Language Models (LALMs) have significantly advanced audio understanding but introduce critical security risks, particularly through audio jailbreaks. While prior work has focused on English-centric attacks, we expose a far more severe vulnerability: adversarial multilingual and multi-accent audio jailbreaks, where linguistic and acoustic variations dramatically amplify attack success. In this paper, we introduce Multi-AudioJail, the first systematic framework to exploit these vulnerabilities through (1) a novel dataset of adversarially perturbed multilingual/multi-accent audio jailbreaking prompts, and (2) a hierarchical evaluation pipeline revealing that how acoustic perturbations (e.g., reverberation, echo, and whisper effects) interacts with cross-lingual phonetics to cause jailbreak success rates (JSRs) to surge by up to +57.25 percentage points (e.g., reverberated Kenyan-accented attack on MERaLiON). Crucially, our work further reveals that multimodal LLMs are inherently more vulnerable than unimodal systems: attackers need only exploit the weakest link (e.g., non-English audio inputs) to compromise the entire model, which we empirically show by multilingual audio-only attacks achieving 3.1x higher success rates than text-only attacks. We plan to release our dataset to spur research into cross-modal defenses, urging the community to address this expanding attack surface in multimodality as LALMs evolve. 

---
# Hard-constraining Neumann boundary conditions in physics-informed neural networks via Fourier feature embeddings 

**Authors**: Christopher Straub, Philipp Brendel, Vlad Medvedev, Andreas Rosskopf  

**Link**: [PDF](https://arxiv.org/pdf/2504.01093)  

**Abstract**: We present a novel approach to hard-constrain Neumann boundary conditions in physics-informed neural networks (PINNs) using Fourier feature embeddings. Neumann boundary conditions are used to described critical processes in various application, yet they are more challenging to hard-constrain in PINNs than Dirichlet conditions. Our method employs specific Fourier feature embeddings to directly incorporate Neumann boundary conditions into the neural network's architecture instead of learning them. The embedding can be naturally extended by high frequency modes to better capture high frequency phenomena. We demonstrate the efficacy of our approach through experiments on a diffusion problem, for which our method outperforms existing hard-constraining methods and classical PINNs, particularly in multiscale and high frequency scenarios. 

---
# HomeEmergency -- Using Audio to Find and Respond to Emergencies in the Home 

**Authors**: James F. Mullen Jr, Dhruva Kumar, Xuewei Qi, Rajasimman Madhivanan, Arnie Sen, Dinesh Manocha, Richard Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.01089)  

**Abstract**: In the United States alone accidental home deaths exceed 128,000 per year. Our work aims to enable home robots who respond to emergency scenarios in the home, preventing injuries and deaths. We introduce a new dataset of household emergencies based in the ThreeDWorld simulator. Each scenario in our dataset begins with an instantaneous or periodic sound which may or may not be an emergency. The agent must navigate the multi-room home scene using prior observations, alongside audio signals and images from the simulator, to determine if there is an emergency or not.
In addition to our new dataset, we present a modular approach for localizing and identifying potential home emergencies. Underpinning our approach is a novel probabilistic dynamic scene graph (P-DSG), where our key insight is that graph nodes corresponding to agents can be represented with a probabilistic edge. This edge, when refined using Bayesian inference, enables efficient and effective localization of agents in the scene. We also utilize multi-modal vision-language models (VLMs) as a component in our approach, determining object traits (e.g. flammability) and identifying emergencies. We present a demonstration of our method completing a real-world version of our task on a consumer robot, showing the transferability of both our task and our method. Our dataset will be released to the public upon this papers publication. 

---
# Knowledge-Base based Semantic Image Transmission Using CLIP 

**Authors**: Chongyang Li, Yanmei He, Tianqian Zhang, Mingjian He, Shouyin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.01053)  

**Abstract**: This paper proposes a novel knowledge-Base (KB) assisted semantic communication framework for image transmission. At the receiver, a Facebook AI Similarity Search (FAISS) based vector database is constructed by extracting semantic embeddings from images using the Contrastive Language-Image Pre-Training (CLIP) model. During transmission, the transmitter first extracts a 512-dimensional semantic feature using the CLIP model, then compresses it with a lightweight neural network for transmission. After receiving the signal, the receiver reconstructs the feature back to 512 dimensions and performs similarity matching from the KB to retrieve the most semantically similar image. Semantic transmission success is determined by category consistency between the transmitted and retrieved images, rather than traditional metrics like Peak Signal-to-Noise Ratio (PSNR). The proposed system prioritizes semantic accuracy, offering a new evaluation paradigm for semantic-aware communication systems. Experimental validation on CIFAR100 demonstrates the effectiveness of the framework in achieving semantic image transmission. 

---
# Predicting Movie Production Years through Facial Recognition of Actors with Machine Learning 

**Authors**: Asraa Muayed Abdalah, Noor Redha Alkazaz  

**Link**: [PDF](https://arxiv.org/pdf/2504.01047)  

**Abstract**: This study used machine learning algorithms to identify actors and extract the age of actors from images taken randomly from movies. The use of images taken from Arab movies includes challenges such as non-uniform lighting, different and multiple poses for the actors and multiple elements with the actor or a group of actors. Additionally, the use of make-up, wigs, beards, and wearing different accessories and costumes made it difficult for the system to identify the personality of the same actor. The Arab Actors Dataset-AAD comprises 574 images sourced from various movies, encompassing both black and white as well as color compositions. The images depict complete scenes or fragments thereof. Multiple models were employed for feature extraction, and diverse machine learning algorithms were utilized during the classification and prediction stages to determine the most effective algorithm for handling such image types. The study demonstrated the effectiveness of the Logistic Regression model exhibited the best performance compared to other models in the training phase, as evidenced by its AUC, precision, CA and F1score values of 99%, 86%, 85.5% and 84.2% respectively. The findings of this study can be used to improve the precision and reliability of facial recognition technology for various uses as with movies search services, movie suggestion algorithms, and genre classification of movies. 

---
# Are clinicians ethically obligated to disclose their use of medical machine learning systems to patients? 

**Authors**: Joshua Hatherley  

**Link**: [PDF](https://arxiv.org/pdf/2504.01043)  

**Abstract**: It is commonly accepted that clinicians are ethically obligated to disclose their use of medical machine learning systems to patients, and that failure to do so would amount to a moral fault for which clinicians ought to be held accountable. Call this "the disclosure thesis." Four main arguments have been, or could be, given to support the disclosure thesis in the ethics literature: the risk-based argument, the rights-based argument, the materiality argument, and the autonomy argument. In this article, I argue that each of these four arguments are unconvincing, and therefore, that the disclosure thesis ought to be rejected. I suggest that mandating disclosure may also even risk harming patients by providing stakeholders with a way to avoid accountability for harm that results from improper applications or uses of these systems. 

---
# One Person, One Bot 

**Authors**: Liat Lavi  

**Link**: [PDF](https://arxiv.org/pdf/2504.01039)  

**Abstract**: This short paper puts forward a vision for a new democratic model enabled by the recent technological advances in agentic AI. It therefore opens with drawing a clear and concise picture of the model, and only later addresses related proposals and research directions, and concerns regarding feasibility and safety. It ends with a note on the timeliness of this idea and on optimism. The model proposed is that of assigning each citizen an AI Agent that would serve as their political delegate, enabling the return to direct democracy. The paper examines this models relation to existing research, its potential setbacks and feasibility and argues for its further development. 

---
# Artificial intelligence and democracy: Towards digital authoritarianism or a democratic upgrade? 

**Authors**: Fereniki Panagopoulou  

**Link**: [PDF](https://arxiv.org/pdf/2504.01034)  

**Abstract**: Do robots vote? Do machines make decisions instead of us? No, (at least not yet), but this is something that could happen. The impact of Artificial Intelligence (AI) on democracy is a complex issue that requires thorough research and careful regulation. At the most important level, that of the electoral process, it is noted that it is not determined by the AI, but it is greatly impacted by its multiple applications. New types of online campaigns, driven by AI applications, are replacing traditional ones. The potential for manipulating voters and indirectly influencing the electoral outcome should not be underestimated. Certainly, instances of voter manipulation are not absent from traditional political campaigns, with the only difference being that digital manipulation is often carried out without our knowledge, e.g. by monitoring our behavior on social media. Nevertheless, we should not overlook the positive impact that AI has in the upgrading of democratic institutions by providing a forum for participation in decision-making. In this context, as a first step, we look into the potential jeopardization of democratic processes posed by the use of AI tools. Secondly, we consider the possibility of strengthening democratic processes by using AI, as well as the democratization of AI itself through the possibilities it offers. And thirdly, the impact of AI on the representative system is also discussed. The paper is concluded with recommendations and conclusions. 

---
# Who Owns the Output? Bridging Law and Technology in LLMs Attribution 

**Authors**: Emanuele Mezzi, Asimina Mertzani, Michael P. Manis, Siyanna Lilova, Nicholas Vadivoulis, Stamatis Gatirdakis, Styliani Roussou, Rodayna Hmede  

**Link**: [PDF](https://arxiv.org/pdf/2504.01032)  

**Abstract**: Since the introduction of ChatGPT in 2022, Large language models (LLMs) and Large Multimodal Models (LMM) have transformed content creation, enabling the generation of human-quality content, spanning every medium, text, images, videos, and audio. The chances offered by generative AI models are endless and are drastically reducing the time required to generate content and usually raising the quality of the generation. However, considering the complexity and the difficult traceability of the generated content, the use of these tools provides challenges in attributing AI-generated content. The difficult attribution resides for a variety of reasons, starting from the lack of a systematic fingerprinting of the generated content and ending with the enormous amount of data on which LLMs and LMM are trained, which makes it difficult to connect generated content to the training data. This scenario is raising concerns about intellectual property and ethical responsibilities. To address these concerns, in this paper, we bridge the technological, ethical, and legislative aspects, by proposing a review of the legislative and technological instruments today available and proposing a legal framework to ensure accountability. In the end, we propose three use cases of how these can be combined to guarantee that attribution is respected. However, even though the techniques available today can guarantee a greater attribution to a greater extent, strong limitations still apply, that can be solved uniquely by the development of new attribution techniques, to be applied to LLMs and LMMs. 

---
# Who is Responsible When AI Fails? Mapping Causes, Entities, and Consequences of AI Privacy and Ethical Incidents 

**Authors**: Hilda Hadan, Reza Hadi Mogavi, Leah Zhang-Kennedy, Lennart E. Nacke  

**Link**: [PDF](https://arxiv.org/pdf/2504.01029)  

**Abstract**: The rapid growth of artificial intelligence (AI) technologies has changed decision-making in many fields. But, it has also raised major privacy and ethical concerns. However, many AI incidents taxonomies and guidelines for academia, industry, and government lack grounding in real-world incidents. We analyzed 202 real-world AI privacy and ethical incidents. This produced a taxonomy that classifies incident types across AI lifecycle stages. It accounts for contextual factors such as causes, responsible entities, disclosure sources, and impacts. Our findings show insufficient incident reporting from AI developers and users. Many incidents are caused by poor organizational decisions and legal non-compliance. Only a few legal actions and corrective measures exist, while risk-mitigation efforts are limited. Our taxonomy contributes a structured approach in reporting of future AI incidents. Our findings demonstrate that current AI governance frameworks are inadequate. We urgently need child-specific protections and AI policies on social media. They must moderate and reduce the spread of harmful AI-generated content. Our research provides insights for policymakers and practitioners, which lets them design ethical AI. It also support AI incident detection and risk management. Finally, it guides AI policy development. Improved policies will protect people from harmful AI applications and support innovation in AI systems. 

---
# Diagnosis of Pulmonary Hypertension by Integrating Multimodal Data with a Hybrid Graph Convolutional and Transformer Network 

**Authors**: Fubao Zhu, Yang Zhang, Gengmin Liang, Jiaofen Nan, Yanting Li, Chuang Han, Danyang Sun, Zhiguo Wang, Chen Zhao, Wenxuan Zhou, Jian He, Yi Xu, Iokfai Cheang, Xu Zhu, Yanli Zhou, Weihua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.01025)  

**Abstract**: Early and accurate diagnosis of pulmonary hypertension (PH) is essential for optimal patient management. Differentiating between pre-capillary and post-capillary PH is critical for guiding treatment decisions. This study develops and validates a deep learning-based diagnostic model for PH, designed to classify patients as non-PH, pre-capillary PH, or post-capillary PH. This retrospective study analyzed data from 204 patients (112 with pre-capillary PH, 32 with post-capillary PH, and 60 non-PH controls) at the First Affiliated Hospital of Nanjing Medical University. Diagnoses were confirmed through right heart catheterization. We selected 6 samples from each category for the test set (18 samples, 10%), with the remaining 186 samples used for the training set. This process was repeated 35 times for testing. This paper proposes a deep learning model that combines Graph convolutional networks (GCN), Convolutional neural networks (CNN), and Transformers. The model was developed to process multimodal data, including short-axis (SAX) sequences, four-chamber (4CH) sequences, and clinical parameters. Our model achieved a performance of Area under the receiver operating characteristic curve (AUC) = 0.81 +- 0.06(standard deviation) and Accuracy (ACC) = 0.73 +- 0.06 on the test set. The discriminative abilities were as follows: non-PH subjects (AUC = 0.74 +- 0.11), pre-capillary PH (AUC = 0.86 +- 0.06), and post-capillary PH (AUC = 0.83 +- 0.10). It has the potential to support clinical decision-making by effectively integrating multimodal data to assist physicians in making accurate and timely diagnoses. 

---
# Gaze-Guided 3D Hand Motion Prediction for Detecting Intent in Egocentric Grasping Tasks 

**Authors**: Yufei He, Xucong Zhang, Arno H. A. Stienen  

**Link**: [PDF](https://arxiv.org/pdf/2504.01024)  

**Abstract**: Human intention detection with hand motion prediction is critical to drive the upper-extremity assistive robots in neurorehabilitation applications. However, the traditional methods relying on physiological signal measurement are restrictive and often lack environmental context. We propose a novel approach that predicts future sequences of both hand poses and joint positions. This method integrates gaze information, historical hand motion sequences, and environmental object data, adapting dynamically to the assistive needs of the patient without prior knowledge of the intended object for grasping. Specifically, we use a vector-quantized variational autoencoder for robust hand pose encoding with an autoregressive generative transformer for effective hand motion sequence prediction. We demonstrate the usability of these novel techniques in a pilot study with healthy subjects. To train and evaluate the proposed method, we collect a dataset consisting of various types of grasp actions on different objects from multiple subjects. Through extensive experiments, we demonstrate that the proposed method can successfully predict sequential hand movement. Especially, the gaze information shows significant enhancements in prediction capabilities, particularly with fewer input frames, highlighting the potential of the proposed method for real-world applications. 

---
