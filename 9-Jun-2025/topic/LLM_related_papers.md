# Can Theoretical Physics Research Benefit from Language Agents? 

**Authors**: Sirui Lu, Zhijing Jin, Terry Jingchen Zhang, Pavel Kos, J. Ignacio Cirac, Bernhard Schölkopf  

**Link**: [PDF](https://arxiv.org/pdf/2506.06214)  

**Abstract**: Large Language Models (LLMs) are rapidly advancing across diverse domains, yet their application in theoretical physics research is not yet mature. This position paper argues that LLM agents can potentially help accelerate theoretical, computational, and applied physics when properly integrated with domain knowledge and toolbox. We analyze current LLM capabilities for physics -- from mathematical reasoning to code generation -- identifying critical gaps in physical intuition, constraint satisfaction, and reliable reasoning. We envision future physics-specialized LLMs that could handle multimodal data, propose testable hypotheses, and design experiments. Realizing this vision requires addressing fundamental challenges: ensuring physical consistency, and developing robust verification methods. We call for collaborative efforts between physics and AI communities to help advance scientific discovery in physics. 

---
# Bridging External and Parametric Knowledge: Mitigating Hallucination of LLMs with Shared-Private Semantic Synergy in Dual-Stream Knowledge 

**Authors**: Yi Sui, Chaozhuo Li, Chen Zhang, Dawei song, Qiuchi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06240)  

**Abstract**: Retrieval-augmented generation (RAG) is a cost-effective approach to mitigate the hallucination of Large Language Models (LLMs) by incorporating the retrieved external knowledge into the generation process. However, external knowledge may conflict with the parametric knowledge of LLMs. Furthermore, current LLMs lack inherent mechanisms for resolving such knowledge conflicts, making traditional RAG methods suffer from degraded performance and stability. Thus, we propose a Dual-Stream Knowledge-Augmented Framework for Shared-Private Semantic Synergy (DSSP-RAG). Central to the framework is a novel approach that refines self-attention into a mixed-attention, distinguishing shared and private semantics for a controlled internal-external knowledge integration. To effectively facilitate DSSP in RAG, we further introduce an unsupervised hallucination detection method based on cognitive uncertainty, ensuring the necessity of introducing knowledge, and an Energy Quotient (EQ) based on attention difference matrices to reduce noise in the retrieved external knowledge. Extensive experiments on benchmark datasets show that DSSP-RAG can effectively resolve conflicts and enhance the complementarity of dual-stream knowledge, leading to superior performance over strong baselines. 

---
# Let's CONFER: A Dataset for Evaluating Natural Language Inference Models on CONditional InFERence and Presupposition 

**Authors**: Tara Azin, Daniel Dumitrescu, Diana Inkpen, Raj Singh  

**Link**: [PDF](https://arxiv.org/pdf/2506.06133)  

**Abstract**: Natural Language Inference (NLI) is the task of determining whether a sentence pair represents entailment, contradiction, or a neutral relationship. While NLI models perform well on many inference tasks, their ability to handle fine-grained pragmatic inferences, particularly presupposition in conditionals, remains underexplored. In this study, we introduce CONFER, a novel dataset designed to evaluate how NLI models process inference in conditional sentences. We assess the performance of four NLI models, including two pre-trained models, to examine their generalization to conditional reasoning. Additionally, we evaluate Large Language Models (LLMs), including GPT-4o, LLaMA, Gemma, and DeepSeek-R1, in zero-shot and few-shot prompting settings to analyze their ability to infer presuppositions with and without prior context. Our findings indicate that NLI models struggle with presuppositional reasoning in conditionals, and fine-tuning on existing NLI datasets does not necessarily improve their performance. 

---
# Zero-Shot Detection of LLM-Generated Code via Approximated Task Conditioning 

**Authors**: Maor Ashkenazi, Ofir Brenner, Tal Furman Shohet, Eran Treister  

**Link**: [PDF](https://arxiv.org/pdf/2506.06069)  

**Abstract**: Detecting Large Language Model (LLM)-generated code is a growing challenge with implications for security, intellectual property, and academic integrity. We investigate the role of conditional probability distributions in improving zero-shot LLM-generated code detection, when considering both the code and the corresponding task prompt that generated it. Our key insight is that when evaluating the probability distribution of code tokens using an LLM, there is little difference between LLM-generated and human-written code. However, conditioning on the task reveals notable differences. This contrasts with natural language text, where differences exist even in the unconditional distributions. Leveraging this, we propose a novel zero-shot detection approach that approximates the original task used to generate a given code snippet and then evaluates token-level entropy under the approximated task conditioning (ATC). We further provide a mathematical intuition, contextualizing our method relative to previous approaches. ATC requires neither access to the generator LLM nor the original task prompts, making it practical for real-world applications. To the best of our knowledge, it achieves state-of-the-art results across benchmarks and generalizes across programming languages, including Python, CPP, and Java. Our findings highlight the importance of task-level conditioning for LLM-generated code detection. The supplementary materials and code are available at this https URL, including the dataset gathering implementation, to foster further research in this area. 

---
# Reinforcing Code Generation: Improving Text-to-SQL with Execution-Based Learning 

**Authors**: Atharv Kulkarni, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.06093)  

**Abstract**: In this work, we study the problem of code generation with a large language model (LLM), with a focus on generating SQL queries from natural language questions. We ask: Instead of using supervised fine tuning with text-code pairs, can we tune a model by having it interact with a database engine? We frame this problem as a reinforcement learning problem where the model receives execution-based feedback from the environment in the form of scalar rewards. These rewards penalize execution failures and assign positive values when a query returns a correct answer. We use the rewards within the Group Relative Policy Optimization (GRPO) framework. We use a tabular reasoning benchmark to test and evaluate our findings. We find that with only weak supervision in the form of question-answer pairs, RL-tuning improves the accuracy of model generated SQL code from 31.49 to 49.83 while reducing error percentage from 25.43% to 14.71%. This improvement allowed the model nearly match the performance performance to the larger SQLCoder-70B model. Our work demonstrates the potential of using execution-based feedback to improve symbolic reasoning capabilities of LLMs. 

---
# Simple Yet Effective: Extracting Private Data Across Clients in Federated Fine-Tuning of Large Language Models 

**Authors**: Yingqi Hu, Zhuo Zhang, Jingyuan Zhang, Lizhen Qu, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06060)  

**Abstract**: Federated fine-tuning of large language models (FedLLMs) presents a promising approach for achieving strong model performance while preserving data privacy in sensitive domains. However, the inherent memorization ability of LLMs makes them vulnerable to training data extraction attacks. To investigate this risk, we introduce simple yet effective extraction attack algorithms specifically designed for FedLLMs. In contrast to prior "verbatim" extraction attacks, which assume access to fragments from all training data, our approach operates under a more realistic threat model, where the attacker only has access to a single client's data and aims to extract previously unseen personally identifiable information (PII) from other clients. This requires leveraging contextual prefixes held by the attacker to generalize across clients. To evaluate the effectiveness of our approaches, we propose two rigorous metrics-coverage rate and efficiency-and extend a real-world legal dataset with PII annotations aligned with CPIS, GDPR, and CCPA standards, achieving 89.9% human-verified precision. Experimental results show that our method can extract up to 56.57% of victim-exclusive PII, with "Address," "Birthday," and "Name" being the most vulnerable categories. Our findings underscore the pressing need for robust defense strategies and contribute a new benchmark and evaluation framework for future research in privacy-preserving federated learning. 

---
# Large Language Models are Demonstration Pre-Selectors for Themselves 

**Authors**: Jiarui Jin, Yuwei Wu, Haoxuan Li, Xiaoting He, Weinan Zhang, Yiming Yang, Yong Yu, Jun Wang, Mengyue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06033)  

**Abstract**: In-context learning (ICL) with large language models (LLMs) delivers strong few-shot performance by choosing few-shot demonstrations from the entire training data. However, existing ICL methods, which rely on similarity or diversity scores to choose demonstrations, incur high computational costs due to repeatedly retrieval from large-scale datasets for each query. To this end, we propose FEEDER (FEw yet Essential Demonstration prE-selectoR), a novel pre-selection framework that identifies a representative subset of demonstrations containing the most representative examples in the training data, tailored to specific LLMs. To construct this subset, we introduce the "sufficiency" and "necessity" metrics in the pre-selection stage and design a tree-based algorithm to identify representative examples efficiently. Once pre-selected, this representative subset can effectively replace the full training data, improving efficiency while maintaining comparable performance in ICL. Additionally, our pre-selected subset also benefits fine-tuning LLMs, where we introduce a bi-level optimization method that enhances training efficiency without sacrificing performance. Experiments with LLMs ranging from 300M to 8B parameters show that FEEDER can reduce training data size by over 20% while maintaining performance and seamlessly integrating with various downstream demonstration selection strategies in ICL. 

---
# Hey, That's My Data! Label-Only Dataset Inference in Large Language Models 

**Authors**: Chen Xiong, Zihao Wang, Rui Zhu, Tsung-Yi Ho, Pin-Yu Chen, Jingwei Xiong, Haixu Tang, Lucila Ohno-Machado  

**Link**: [PDF](https://arxiv.org/pdf/2506.06057)  

**Abstract**: Large Language Models (LLMs) have revolutionized Natural Language Processing by excelling at interpreting, reasoning about, and generating human language. However, their reliance on large-scale, often proprietary datasets poses a critical challenge: unauthorized usage of such data can lead to copyright infringement and significant financial harm. Existing dataset-inference methods typically depend on log probabilities to detect suspicious training material, yet many leading LLMs have begun withholding or obfuscating these signals. This reality underscores the pressing need for label-only approaches capable of identifying dataset membership without relying on internal model logits.
We address this gap by introducing CatShift, a label-only dataset-inference framework that capitalizes on catastrophic forgetting: the tendency of an LLM to overwrite previously learned knowledge when exposed to new data. If a suspicious dataset was previously seen by the model, fine-tuning on a portion of it triggers a pronounced post-tuning shift in the model's outputs; conversely, truly novel data elicits more modest changes. By comparing the model's output shifts for a suspicious dataset against those for a known non-member validation set, we statistically determine whether the suspicious set is likely to have been part of the model's original training corpus. Extensive experiments on both open-source and API-based LLMs validate CatShift's effectiveness in logit-inaccessible settings, offering a robust and practical solution for safeguarding proprietary data. 

---
# AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search 

**Authors**: Yu Li, Lehui Li, Zhihao Wu, Qingmin Liao, Jianye Hao, Kun Shao, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06017)  

**Abstract**: Large language model (LLM) agents have demonstrated strong capabilities across diverse domains. However, designing high-performing agentic systems remains challenging. Existing agent search methods suffer from three major limitations: (1) an emphasis on optimizing agentic workflows while under-utilizing proven human-designed components such as memory, planning, and tool use; (2) high evaluation costs, as each newly generated agent must be fully evaluated on benchmarks; and (3) inefficient search in large search space. In this work, we introduce a comprehensive framework to address these challenges. First, We propose a hierarchical search space that jointly models agentic workflow and composable functional components, enabling richer agentic system designs. Building on this structured design space, we introduce a predictive value model that estimates agent performance given agentic system and task description, allowing for efficient, low-cost evaluation during the search process. Finally, we present a hierarchical Monte Carlo Tree Search (MCTS) strategy informed by uncertainty to guide the search. Experiments on seven benchmarks, covering embodied, math, web, tool, and game, show that our method achieves an average performance gain of 8.34\% over state-of-the-art baselines and exhibits faster search progress with steeper improvement trajectories. Code repo is available at this https URL. 

---
# Token Signature: Predicting Chain-of-Thought Gains with Token Decoding Feature in Large Language Models 

**Authors**: Peijie Liu, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06008)  

**Abstract**: Chain-of-Thought (CoT) technique has proven effective in improving the performance of large language models (LLMs) on complex reasoning tasks. However, the performance gains are inconsistent across different tasks, and the underlying mechanism remains a long-standing research question. In this work, we make a preliminary observation that the monotonicity of token probability distributions may be correlated with the gains achieved through CoT reasoning. Leveraging this insight, we propose two indicators based on the token probability distribution to assess CoT effectiveness across different tasks. By combining instance-level indicators with logistic regression model, we introduce Dynamic CoT, a method that dynamically select between CoT and direct answer. Furthermore, we extend Dynamic CoT to closed-source models by transferring decision strategies learned from open-source models. Our indicators for assessing CoT effectiveness achieve an accuracy of 89.2\%, and Dynamic CoT reduces token consumption by more than 35\% while maintaining high accuracy. Overall, our work offers a novel perspective on the underlying mechanisms of CoT reasoning and provides a framework for its more efficient deployment. 

---
# A Culturally-Rich Romanian NLP Dataset from "Who Wants to Be a Millionaire?" Videos 

**Authors**: Alexandru-Gabriel Ganea, Antonia-Adelina Popovici, Adrian-Marius Dumitran  

**Link**: [PDF](https://arxiv.org/pdf/2506.05991)  

**Abstract**: Large Language Models (LLMs) demonstrate varying performance across languages and cultural contexts. This study introduces a novel, culturally-rich, multilingual dataset derived from video recordings of the Romanian game show "Who Wants to Be a Millionaire?" (Vrei să fii Milionar?). We employed an innovative process combining optical character recognition (OCR), automated text extraction, and manual verification to collect question-answer pairs, enriching them with metadata including question domain (e.g., biology, history), cultural relevance (Romanian-specific vs. international), and difficulty. Benchmarking state-of-the-art LLMs, including Romanian-adapted models, on this dataset revealed significant performance disparities: models consistently achieve higher accuracy (80-95%) on international questions compared to Romanian-specific cultural questions (50-75%). We further investigate these differences through experiments involving machine translation of Romanian questions into English and cross-lingual tests using a comparable dataset in French. Our findings underscore the impact of cultural context and data source on LLM performance and offer practical insights for building robust, culturally-aware multilingual NLP systems, especially in educational domains. The dataset is publicly available at Hugging Face. 

---
# MATP-BENCH: Can MLLM Be a Good Automated Theorem Prover for Multimodal Problems? 

**Authors**: Zhitao He, Zongwei Lyu, Dazhong Chen, Dadi Guo, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2506.06034)  

**Abstract**: Numerous theorems, such as those in geometry, are often presented in multimodal forms (e.g., diagrams). Humans benefit from visual reasoning in such settings, using diagrams to gain intuition and guide the proof process. Modern Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in solving a wide range of mathematical problems. However, the potential of MLLMs as Automated Theorem Provers (ATPs), specifically in the multimodal domain, remains underexplored. In this paper, we introduce the Multimodal Automated Theorem Proving benchmark (MATP-BENCH), a new Multimodal, Multi-level, and Multi-language benchmark designed to evaluate MLLMs in this role as multimodal automated theorem provers. MATP-BENCH consists of 1056 multimodal theorems drawn from high school, university, and competition-level mathematics. All these multimodal problems are accompanied by formalizations in Lean 4, Coq and Isabelle, thus making the benchmark compatible with a wide range of theorem-proving frameworks. MATP-BENCH requires models to integrate sophisticated visual understanding with mastery of a broad spectrum of mathematical knowledge and rigorous symbolic reasoning to generate formal proofs. We use MATP-BENCH to evaluate a variety of advanced multimodal language models. Existing methods can only solve a limited number of the MATP-BENCH problems, indicating that this benchmark poses an open challenge for research on automated theorem proving. 

---
# Let's Put Ourselves in Sally's Shoes: Shoes-of-Others Prefixing Improves Theory of Mind in Large Language Models 

**Authors**: Kazutoshi Shinoda, Nobukatsu Hojo, Kyosuke Nishida, Yoshihiro Yamazaki, Keita Suzuki, Hiroaki Sugiyama, Kuniko Saito  

**Link**: [PDF](https://arxiv.org/pdf/2506.05970)  

**Abstract**: Recent studies have shown that Theory of Mind (ToM) in large language models (LLMs) has not reached human-level performance yet. Since fine-tuning LLMs on ToM datasets often degrades their generalization, several inference-time methods have been proposed to enhance ToM in LLMs. However, existing inference-time methods for ToM are specialized for inferring beliefs from contexts involving changes in the world state. In this study, we present a new inference-time method for ToM, Shoes-of-Others (SoO) prefixing, which makes fewer assumptions about contexts and is applicable to broader scenarios. SoO prefixing simply specifies the beginning of LLM outputs with ``Let's put ourselves in A's shoes.'', where A denotes the target character's name. We evaluate SoO prefixing on two benchmarks that assess ToM in conversational and narrative contexts without changes in the world state and find that it consistently improves ToM across five categories of mental states. Our analysis suggests that SoO prefixing elicits faithful thoughts, thereby improving the ToM performance. 

---
# Elementary Math Word Problem Generation using Large Language Models 

**Authors**: Nimesh Ariyarathne, Harshani Bandara, Yasith Heshan, Omega Gamage, Surangika Ranathunga, Dilan Nayanajith, Yutharsan Sivapalan, Gayathri Lihinikaduarachchi, Tharoosha Vihidun, Meenambika Chandirakumar, Sanujen Premakumar, Sanjula Gathsara  

**Link**: [PDF](https://arxiv.org/pdf/2506.05950)  

**Abstract**: Mathematics is often perceived as a complex subject by students, leading to high failure rates in exams. To improve Mathematics skills, it is important to provide sample questions for students to practice problem-solving. Manually creating Math Word Problems (MWPs) is time consuming for tutors, because they have to type in natural language while adhering to grammar and spelling rules of the language. Existing Deep Learning techniques for MWP generation either require a tutor to provide the initial portion of the MWP, and/or additional information such as an equation. In this paper, we present an MWP generation system based on Large Language Models (LLMs) that overcome the need for additional input - the only input to our system is the number of MWPs needed, the grade and the type of question (e.g. addition, subtraction). Unlike the existing LLM-based solutions for MWP generation, we carried out an extensive set of experiments involving different LLMs, prompting strategies, techniques to improve the diversity of questions, as well as techniques that employ human feedback to improve LLM performance. Human and automated evaluations confirmed that the generated MWPs are high in quality, with minimal spelling and grammar issues. However, LLMs still struggle to generate questions that adhere to the specified grade and question type requirements. 

---
# DynamicMind: A Tri-Mode Thinking System for Large Language Models 

**Authors**: Wei Li, Yanbin Wei, Qiushi Huang, Jiangyue Yan, Yang Chen, James T. Kwok, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05936)  

**Abstract**: Modern large language models (LLMs) often struggle to dynamically adapt their reasoning depth to varying task complexities, leading to suboptimal performance or inefficient resource utilization. To address this, we introduce DynamicMind, a novel tri-mode thinking system. DynamicMind empowers LLMs to autonomously select between Fast, Normal, and Slow thinking modes for zero-shot question answering (ZSQA) tasks through cognitive-inspired prompt engineering. Our framework's core innovations include: (1) expanding the established dual-process framework of fast and slow thinking into a tri-mode thinking system involving a normal thinking mode to preserve the intrinsic capabilities of LLM; (2) proposing the Thinking Density metric, which aligns computational resource allocation with problem complexity; and (3) developing the Thinking Mode Capacity (TMC) dataset and a lightweight Mind Router to predict the optimal thinking mode. Extensive experiments across diverse mathematical, commonsense, and scientific QA benchmarks demonstrate that DynamicMind achieves superior ZSQA capabilities while establishing an effective trade-off between performance and computational efficiency. 

---
# Does It Run and Is That Enough? Revisiting Text-to-Chart Generation with a Multi-Agent Approach 

**Authors**: James Ford, Anthony Rios  

**Link**: [PDF](https://arxiv.org/pdf/2506.06175)  

**Abstract**: Large language models can translate natural-language chart descriptions into runnable code, yet approximately 15\% of the generated scripts still fail to execute, even after supervised fine-tuning and reinforcement learning. We investigate whether this persistent error rate stems from model limitations or from reliance on a single-prompt design. To explore this, we propose a lightweight multi-agent pipeline that separates drafting, execution, repair, and judgment, using only an off-the-shelf GPT-4o-mini model. On the \textsc{Text2Chart31} benchmark, our system reduces execution errors to 4.5\% within three repair iterations, outperforming the strongest fine-tuned baseline by nearly 5 percentage points while requiring significantly less compute. Similar performance is observed on the \textsc{ChartX} benchmark, with an error rate of 4.6\%, demonstrating strong generalization. Under current benchmarks, execution success appears largely solved. However, manual review reveals that 6 out of 100 sampled charts contain hallucinations, and an LLM-based accessibility audit shows that only 33.3\% (\textsc{Text2Chart31}) and 7.2\% (\textsc{ChartX}) of generated charts satisfy basic colorblindness guidelines. These findings suggest that future work should shift focus from execution reliability toward improving chart aesthetics, semantic fidelity, and accessibility. 

---
# Generating Grounded Responses to Counter Misinformation via Learning Efficient Fine-Grained Critiques 

**Authors**: Xiaofei Xu, Xiuzhen Zhang, Ke Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.05924)  

**Abstract**: Fake news and misinformation poses a significant threat to society, making efficient mitigation essential. However, manual fact-checking is costly and lacks scalability. Large Language Models (LLMs) offer promise in automating counter-response generation to mitigate misinformation, but a critical challenge lies in their tendency to hallucinate non-factual information. Existing models mainly rely on LLM self-feedback to reduce hallucination, but this approach is computationally expensive. In this paper, we propose MisMitiFact, Misinformation Mitigation grounded in Facts, an efficient framework for generating fact-grounded counter-responses at scale. MisMitiFact generates simple critique feedback to refine LLM outputs, ensuring responses are grounded in evidence. We develop lightweight, fine-grained critique models trained on data sourced from readily available fact-checking sites to identify and correct errors in key elements such as numerals, entities, and topics in LLM generations. Experiments show that MisMitiFact generates counter-responses of comparable quality to LLMs' self-feedback while using significantly smaller critique models. Importantly, it achieves ~5x increase in feedback generation throughput, making it highly suitable for cost-effective, large-scale misinformation mitigation. Code and LLM prompt templates are at this https URL. 

---
# MIRIAD: Augmenting LLMs with millions of medical query-response pairs 

**Authors**: Qinyue Zheng, Salman Abdullah, Sam Rawal, Cyril Zakka, Sophie Ostmeier, Maximilian Purk, Eduardo Reis, Eric J. Topol, Jure Leskovec, Michael Moor  

**Link**: [PDF](https://arxiv.org/pdf/2506.06091)  

**Abstract**: LLMs are bound to transform healthcare with advanced decision support and flexible chat assistants. However, LLMs are prone to generate inaccurate medical content. To ground LLMs in high-quality medical knowledge, LLMs have been equipped with external knowledge via RAG, where unstructured medical knowledge is split into small text chunks that can be selectively retrieved and integrated into the LLMs context. Yet, existing RAG pipelines rely on raw, unstructured medical text, which can be noisy, uncurated and difficult for LLMs to effectively leverage. Systematic approaches to organize medical knowledge to best surface it to LLMs are generally lacking. To address these challenges, we introduce MIRIAD, a large-scale, curated corpus of 5,821,948 medical QA pairs, each rephrased from and grounded in a passage from peer-reviewed medical literature using a semi-automated pipeline combining LLM generation, filtering, grounding, and human annotation. Unlike prior medical corpora, which rely on unstructured text, MIRIAD encapsulates web-scale medical knowledge in an operationalized query-response format, which enables more targeted retrieval. Experiments on challenging medical QA benchmarks show that augmenting LLMs with MIRIAD improves accuracy up to 6.7% compared to unstructured RAG baselines with the same source corpus and with the same amount of retrieved text. Moreover, MIRIAD improved the ability of LLMs to detect medical hallucinations by 22.5 to 37% (increase in F1 score). We further introduce MIRIAD-Atlas, an interactive map of MIRIAD spanning 56 medical disciplines, enabling clinical users to visually explore, search, and refine medical knowledge. MIRIAD promises to unlock a wealth of down-stream applications, including medical information retrievers, enhanced RAG applications, and knowledge-grounded chat interfaces, which ultimately enables more reliable LLM applications in healthcare. 

---
# Route-and-Reason: Scaling Large Language Model Reasoning with Reinforced Model Router 

**Authors**: Chenyang Shao, Xinyang Liu, Yutang Lin, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05901)  

**Abstract**: Multi-step reasoning has proven essential for enhancing the problem-solving capabilities of Large Language Models (LLMs) by decomposing complex tasks into intermediate steps, either explicitly or implicitly. Extending the reasoning chain at test time through deeper thought processes or broader exploration, can furthur improve performance, but often incurs substantial costs due to the explosion in token usage. Yet, many reasoning steps are relatively simple and can be handled by more efficient smaller-scale language models (SLMs). This motivates hybrid approaches that allocate subtasks across models of varying capacities. However, realizing such collaboration requires accurate task decomposition and difficulty-aware subtask allocation, which is challenging. To address this, we propose R2-Reasoner, a novel framework that enables collaborative reasoning across heterogeneous LLMs by dynamically routing sub-tasks based on estimated complexity. At the core of our framework is a Reinforced Model Router, composed of a task decomposer and a subtask allocator. The task decomposer segments complex input queries into logically ordered subtasks, while the subtask allocator assigns each subtask to the most appropriate model, ranging from lightweight SLMs to powerful LLMs, balancing accuracy and efficiency. To train this router, we introduce a staged pipeline that combines supervised fine-tuning on task-specific datasets with Group Relative Policy Optimization algorithm, enabling self-supervised refinement through iterative reinforcement learning. Extensive experiments across four challenging benchmarks demonstrate that R2-Reasoner reduces API costs by 86.85% while maintaining or surpassing baseline accuracy. Our framework paves the way for more cost-effective and adaptive LLM reasoning. The code is open-source at this https URL . 

---
# Discrete Minds in a Continuous World: Do Language Models Know Time Passes? 

**Authors**: Minghan Wang, Ye Bai, Thuy-Trang Vu, Ehsan Shareghi, Gholamreza Haffari  

**Link**: [PDF](https://arxiv.org/pdf/2506.05790)  

**Abstract**: While Large Language Models (LLMs) excel at temporal reasoning tasks like event ordering and duration estimation, their ability to perceive the actual passage of time remains unexplored. We investigate whether LLMs perceive the passage of time and adapt their decision-making accordingly through three complementary experiments. First, we introduce the Token-Time Hypothesis, positing that LLMs can map discrete token counts to continuous wall-clock time, and validate this through a dialogue duration judgment task. Second, we demonstrate that LLMs could use this awareness to adapt their response length while maintaining accuracy when users express urgency in question answering tasks. Finally, we develop BombRush, an interactive navigation challenge that examines how LLMs modify behavior under progressive time pressure in dynamic environments. Our findings indicate that LLMs possess certain awareness of time passage, enabling them to bridge discrete linguistic tokens and continuous physical time, though this capability varies with model size and reasoning abilities. This work establishes a theoretical foundation for enhancing temporal awareness in LLMs for time-sensitive applications. 

---
# When to Trust Context: Self-Reflective Debates for Context Reliability 

**Authors**: Zeqi Zhou, Fang Wu, Shayan Talaei, Haokai Zhao, Cheng Meixin, Tinson Xu, Amin Saberi, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06020)  

**Abstract**: Large language models frequently encounter conflicts between their parametric knowledge and contextual input, often resulting in factual inconsistencies or hallucinations. We propose Self-Reflective Debate for Contextual Reliability (SR-DCR), a lightweight framework that integrates token-level self-confidence with an asymmetric multi-agent debate to adjudicate such conflicts. A critic, deprived of context, challenges a defender who argues from the given passage; a judge model evaluates the debate and determines the context's reliability. The final answer is selected by combining the verdict with model confidence. Experiments on the ClashEval benchmark demonstrate that SR-DCR consistently enhances robustness to misleading context while maintaining accuracy on trustworthy inputs, outperforming both classical debate and confidence-only baselines with minimal computational overhead. The code is available at this https URL. 

---
# Unlocking Recursive Thinking of LLMs: Alignment via Refinement 

**Authors**: Haoke Zhang, Xiaobo Liang, Cunxiang Wang, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06009)  

**Abstract**: The OpenAI o1-series models have demonstrated that leveraging long-form Chain of Thought (CoT) can substantially enhance performance. However, the recursive thinking capabilities of Large Language Models (LLMs) remain limited, particularly in the absence of expert-curated data for distillation. In this paper, we propose \textbf{AvR}: \textbf{Alignment via Refinement}, a novel method aimed at unlocking the potential of LLMs for recursive reasoning through long-form CoT. AvR introduces a refinement process that integrates criticism and improvement actions, guided by differentiable learning techniques to optimize \textbf{refinement-aware rewards}. As a result, the synthesized multi-round data can be organized as a long refinement thought, further enabling test-time scaling. Experimental results show that AvR significantly outperforms conventional preference optimization methods. Notably, with only 3k synthetic samples, our method boosts the performance of the LLaMA-3-8B-Instruct model by over 20\% in win rate on AlpacaEval 2.0. Our code is available at Github (this https URL). 

---
# BioMol-MQA: A Multi-Modal Question Answering Dataset For LLM Reasoning Over Bio-Molecular Interactions 

**Authors**: Saptarshi Sengupta, Shuhua Yang, Paul Kwong Yu, Fali Wang, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05766)  

**Abstract**: Retrieval augmented generation (RAG) has shown great power in improving Large Language Models (LLMs). However, most existing RAG-based LLMs are dedicated to retrieving single modality information, mainly text; while for many real-world problems, such as healthcare, information relevant to queries can manifest in various modalities such as knowledge graph, text (clinical notes), and complex molecular structure. Thus, being able to retrieve relevant multi-modality domain-specific information, and reason and synthesize diverse knowledge to generate an accurate response is important. To address the gap, we present BioMol-MQA, a new question-answering (QA) dataset on polypharmacy, which is composed of two parts (i) a multimodal knowledge graph (KG) with text and molecular structure for information retrieval; and (ii) challenging questions that designed to test LLM capabilities in retrieving and reasoning over multimodal KG to answer questions. Our benchmarks indicate that existing LLMs struggle to answer these questions and do well only when given the necessary background data, signaling the necessity for strong RAG frameworks. 

---
# Do LLMs Really Forget? Evaluating Unlearning with Knowledge Correlation and Confidence Awareness 

**Authors**: Rongzhe Wei, Peizhi Niu, Hans Hao-Hsun Hsu, Ruihan Wu, Haoteng Yin, Mohsen Ghassemi, Yifan Li, Vamsi K. Potluru, Eli Chien, Kamalika Chaudhuri, Olgica Milenkovic, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05735)  

**Abstract**: Machine unlearning techniques aim to mitigate unintended memorization in large language models (LLMs). However, existing approaches predominantly focus on the explicit removal of isolated facts, often overlooking latent inferential dependencies and the non-deterministic nature of knowledge within LLMs. Consequently, facts presumed forgotten may persist implicitly through correlated information. To address these challenges, we propose a knowledge unlearning evaluation framework that more accurately captures the implicit structure of real-world knowledge by representing relevant factual contexts as knowledge graphs with associated confidence scores. We further develop an inference-based evaluation protocol leveraging powerful LLMs as judges; these judges reason over the extracted knowledge subgraph to determine unlearning success. Our LLM judges utilize carefully designed prompts and are calibrated against human evaluations to ensure their trustworthiness and stability. Extensive experiments on our newly constructed benchmark demonstrate that our framework provides a more realistic and rigorous assessment of unlearning performance. Moreover, our findings reveal that current evaluation strategies tend to overestimate unlearning effectiveness. Our code is publicly available at this https URL. 

---
# LLM-Symbolic Integration for Robust Temporal Tabular Reasoning 

**Authors**: Atharv Kulkarni, Kushagra Dixit, Vivek Srikumar, Dan Roth, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.05746)  

**Abstract**: Temporal tabular question answering presents a significant challenge for Large Language Models (LLMs), requiring robust reasoning over structured data, which is a task where traditional prompting methods often fall short. These methods face challenges such as memorization, sensitivity to table size, and reduced performance on complex queries. To overcome these limitations, we introduce TempTabQA-C, a synthetic dataset designed for systematic and controlled evaluations, alongside a symbolic intermediate representation that transforms tables into database schemas. This structured approach allows LLMs to generate and execute SQL queries, enhancing generalization and mitigating biases. By incorporating adaptive few-shot prompting with contextually tailored examples, our method achieves superior robustness, scalability, and performance. Experimental results consistently highlight improvements across key challenges, setting a new benchmark for robust temporal reasoning with LLMs. 

---
# Large Language Models are Good Relational Learners 

**Authors**: Fang Wu, Vijay Prakash Dwivedi, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2506.05725)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various domains, yet their application to relational deep learning (RDL) remains underexplored. Existing approaches adapt LLMs by traversing relational links between entities in a database and converting the structured data into flat text documents. Still, this text-based serialization disregards critical relational structures, introduces redundancy, and often exceeds standard LLM context lengths. We introduce Rel-LLM, a novel architecture that utilizes a graph neural network (GNN)- based encoder to generate structured relational prompts for LLMs within a retrieval-augmented generation (RAG) framework. Unlike traditional text-based serialization approaches, our method preserves the inherent relational structure of databases while enabling LLMs to effectively process and reason over complex entity relationships. Specifically, the GNN encoder extracts a local subgraph around an entity to build feature representations that contain relevant entity relationships and temporal dependencies. These representations are transformed into structured prompts using a denormalization process, effectively allowing the LLM to reason over relational structures. Through extensive experiments, we demonstrate that Rel-LLM outperforms existing methods on key RDL tasks, offering a scalable and efficient approach to integrating LLMs with structured data sources. Code is available at this https URL. 

---
# RKEFino1: A Regulation Knowledge-Enhanced Large Language Model 

**Authors**: Yan Wang, Yueru He, Ruoyu Xiang, Jeff Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.05700)  

**Abstract**: Recent advances in large language models (LLMs) hold great promise for financial applications but introduce critical accuracy and compliance challenges in Digital Regulatory Reporting (DRR). To address these issues, we propose RKEFino1, a regulation knowledge-enhanced financial reasoning model built upon Fino1, fine-tuned with domain knowledge from XBRL, CDM, and MOF. We formulate two QA tasks-knowledge-based and mathematical reasoning-and introduce a novel Numerical NER task covering financial entities in both sentences and tables. Experimental results demonstrate the effectiveness and generalization capacity of RKEFino1 in compliance-critical financial tasks. We have released our model on Hugging Face. 

---
# Zero-Shot Event Causality Identification via Multi-source Evidence Fuzzy Aggregation with Large Language Models 

**Authors**: Zefan Zeng, Xingchen Hu, Qing Cheng, Weiping Ding, Wentao Li, Zhong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05675)  

**Abstract**: Event Causality Identification (ECI) aims to detect causal relationships between events in textual contexts. Existing ECI models predominantly rely on supervised methodologies, suffering from dependence on large-scale annotated data. Although Large Language Models (LLMs) enable zero-shot ECI, they are prone to causal hallucination-erroneously establishing spurious causal links. To address these challenges, we propose MEFA, a novel zero-shot framework based on Multi-source Evidence Fuzzy Aggregation. First, we decompose causality reasoning into three main tasks (temporality determination, necessity analysis, and sufficiency verification) complemented by three auxiliary tasks. Second, leveraging meticulously designed prompts, we guide LLMs to generate uncertain responses and deterministic outputs. Finally, we quantify LLM's responses of sub-tasks and employ fuzzy aggregation to integrate these evidence for causality scoring and causality determination. Extensive experiments on three benchmarks demonstrate that MEFA outperforms second-best unsupervised baselines by 6.2% in F1-score and 9.3% in precision, while significantly reducing hallucination-induced errors. In-depth analysis verify the effectiveness of task decomposition and the superiority of fuzzy aggregation. 

---
# Cross-lingual Collapse: How Language-Centric Foundation Models Shape Reasoning in Large Language Models 

**Authors**: Cheonbok Park, Jeonghoon Kim, Joosung Lee, Sanghwan Bae, Jaegul Choo, Kangmin Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2506.05850)  

**Abstract**: We identify \textbf{Cross-lingual Collapse}, a systematic drift in which the chain-of-thought (CoT) of a multilingual language model reverts to its dominant pre-training language even when the prompt is expressed in a different language. Recent large language models (LLMs) with reinforcement learning with verifiable reward (RLVR) have achieved strong logical reasoning performances by exposing their intermediate reasoning traces, giving rise to large reasoning models (LRMs). However, the mechanism behind multilingual reasoning in LRMs is not yet fully explored. To investigate the issue, we fine-tune multilingual LRMs with Group-Relative Policy Optimization (GRPO) on translated versions of the GSM$8$K and SimpleRL-Zoo datasets in three different languages: Chinese, Korean, and Ukrainian. During training, we monitor both task accuracy and language consistency of the reasoning chains. Our experiments reveal three key findings: (i) GRPO rapidly amplifies pre-training language imbalances, leading to the erosion of low-resource languages within just a few hundred updates; (ii) language consistency reward mitigates this drift but does so at the expense of an almost 5 - 10 pp drop in accuracy. and (iii) the resulting language collapse is severely damaging and largely irreversible, as subsequent fine-tuning struggles to steer the model back toward its original target-language reasoning capabilities. Together, these findings point to a remarkable conclusion: \textit{not all languages are trained equally for reasoning}. Furthermore, our paper sheds light on the roles of reward shaping, data difficulty, and pre-training priors in eliciting multilingual reasoning. 

---
# LTG at SemEval-2025 Task 10: Optimizing Context for Classification of Narrative Roles 

**Authors**: Egil Rønningstad, Gaurav Negi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05976)  

**Abstract**: Our contribution to the SemEval 2025 shared task 10, subtask 1 on entity framing, tackles the challenge of providing the necessary segments from longer documents as context for classification with a masked language model. We show that a simple entity-oriented heuristics for context selection can enable text classification using models with limited context window. Our context selection approach and the XLM-RoBERTa language model is on par with, or outperforms, Supervised Fine-Tuning with larger generative language models. 

---
# Can LLMs Express Personality Across Cultures? Introducing CulturalPersonas for Evaluating Trait Alignment 

**Authors**: Priyanka Dey, Yugal Khanter, Aayush Bothra, Jieyu Zhao, Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2506.05670)  

**Abstract**: As LLMs become central to interactive applications, ranging from tutoring to mental health, the ability to express personality in culturally appropriate ways is increasingly important. While recent works have explored personality evaluation of LLMs, they largely overlook the interplay between culture and personality. To address this, we introduce CulturalPersonas, the first large-scale benchmark with human validation for evaluating LLMs' personality expression in culturally grounded, behaviorally rich contexts. Our dataset spans 3,000 scenario-based questions across six diverse countries, designed to elicit personality through everyday scenarios rooted in local values. We evaluate three LLMs, using both multiple-choice and open-ended response formats. Our results show that CulturalPersonas improves alignment with country-specific human personality distributions (over a 20% reduction in Wasserstein distance across models and countries) and elicits more expressive, culturally coherent outputs compared to existing benchmarks. CulturalPersonas surfaces meaningful modulated trait outputs in response to culturally grounded prompts, offering new directions for aligning LLMs to global norms of behavior. By bridging personality expression and cultural nuance, we envision that CulturalPersonas will pave the way for more socially intelligent and globally adaptive LLMs. 

---
# OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation 

**Authors**: Ziyi Wang, Yuxuan Lu, Wenbo Li, Amirali Amini, Bo Sun, Yakov Bart, Weimin Lyu, Jiri Gesi, Tian Wang, Jing Huang, Yu Su, Upol Ehsan, Malihe Alikhani, Toby Jia-Jun Li, Lydia Chilton, Dakuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05606)  

**Abstract**: Can large language models (LLMs) accurately simulate the next web action of a specific user? While LLMs have shown promising capabilities in generating ``believable'' human behaviors, evaluating their ability to mimic real user behaviors remains an open challenge, largely due to the lack of high-quality, publicly available datasets that capture both the observable actions and the internal reasoning of an actual human user. To address this gap, we introduce OPERA, a novel dataset of Observation, Persona, Rationale, and Action collected from real human participants during online shopping sessions. OPERA is the first public dataset that comprehensively captures: user personas, browser observations, fine-grained web actions, and self-reported just-in-time rationales. We developed both an online questionnaire and a custom browser plugin to gather this dataset with high fidelity. Using OPERA, we establish the first benchmark to evaluate how well current LLMs can predict a specific user's next action and rationale with a given persona and <observation, action, rationale> history. This dataset lays the groundwork for future research into LLM agents that aim to act as personalized digital twins for human. 

---
# IYKYK: Using language models to decode extremist cryptolects 

**Authors**: Christine de Kock, Arij Riabi, Zeerak Talat, Michael Sejr Schlichtkrull, Pranava Madhyastha, Ed Hovy  

**Link**: [PDF](https://arxiv.org/pdf/2506.05635)  

**Abstract**: Extremist groups develop complex in-group language, also referred to as cryptolects, to exclude or mislead outsiders. We investigate the ability of current language technologies to detect and interpret the cryptolects of two online extremist platforms. Evaluating eight models across six tasks, our results indicate that general purpose LLMs cannot consistently detect or decode extremist language. However, performance can be significantly improved by domain adaptation and specialised prompting techniques. These results provide important insights to inform the development and deployment of automated moderation technologies. We further develop and release novel labelled and unlabelled datasets, including 19.4M posts from extremist platforms and lexicons validated by human experts. 

---
# Writing-RL: Advancing Long-form Writing via Adaptive Curriculum Reinforcement Learning 

**Authors**: Xuanyu Lei, Chenliang Li, Yuning Wu, Kaiming Liu, Weizhou Shen, Peng Li, Ming Yan, Ji Zhang, Fei Huang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05760)  

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled strong performance in long-form writing, yet existing supervised fine-tuning (SFT) approaches suffer from limitations such as data saturation and restricted learning capacity bounded by teacher signals. In this work, we present Writing-RL: an Adaptive Curriculum Reinforcement Learning framework to advance long-form writing capabilities beyond SFT. The framework consists of three key components: Margin-aware Data Selection strategy that prioritizes samples with high learning potential, Pairwise Comparison Reward mechanism that provides discriminative learning signals in the absence of verifiable rewards, and Dynamic Reference Scheduling approach, which plays a particularly critical role by adaptively adjusting task difficulty based on evolving model performance. Experiments on 7B-scale writer models show that our RL framework largely improves long-form writing performance over strong SFT baselines. Furthermore, we observe that models trained with long-output RL generalize surprisingly well to long-input reasoning tasks, potentially offering a promising perspective for rethinking long-context training. 

---
# SynthesizeMe! Inducing Persona-Guided Prompts for Personalized Reward Models in LLMs 

**Authors**: Michael J Ryan, Omar Shaikh, Aditri Bhagirath, Daniel Frees, William Held, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05598)  

**Abstract**: Recent calls for pluralistic alignment of Large Language Models (LLMs) encourage adapting models to diverse user preferences. However, most prior work on personalized reward models heavily rely on additional identity information, such as demographic details or a predefined set of preference categories. To this end, we introduce SynthesizeMe, an approach to inducing synthetic user personas from user interactions for personalized reward modeling. SynthesizeMe first generates and verifies reasoning to explain user preferences, then induces synthetic user personas from that reasoning, and finally filters to informative prior user interactions in order to build personalized prompts for a particular user. We show that using SynthesizeMe induced prompts improves personalized LLM-as-a-judge accuracy by 4.4% on Chatbot Arena. Combining SynthesizeMe derived prompts with a reward model achieves top performance on PersonalRewardBench: a new curation of user-stratified interactions with chatbots collected from 854 users of Chatbot Arena and PRISM. 

---
# UTSA-NLP at ArchEHR-QA 2025: Improving EHR Question Answering via Self-Consistency Prompting 

**Authors**: Sara Shields-Menard, Zach Reimers, Joshua Gardner, David Perry, Anthony Rios  

**Link**: [PDF](https://arxiv.org/pdf/2506.05589)  

**Abstract**: We describe our system for the ArchEHR-QA Shared Task on answering clinical questions using electronic health records (EHRs). Our approach uses large language models in two steps: first, to find sentences in the EHR relevant to a clinician's question, and second, to generate a short, citation-supported response based on those sentences. We use few-shot prompting, self-consistency, and thresholding to improve the sentence classification step to decide which sentences are essential. We compare several models and find that a smaller 8B model performs better than a larger 70B model for identifying relevant information. Our results show that accurate sentence selection is critical for generating high-quality responses and that self-consistency with thresholding helps make these decisions more reliable. 

---
# MLLM-CL: Continual Learning for Multimodal Large Language Models 

**Authors**: Hongbo Zhao, Fei Zhu, Rundong Wang, Gaofeng Meng, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05453)  

**Abstract**: Recent Multimodal Large Language Models (MLLMs) excel in vision-language understanding but face challenges in adapting to dynamic real-world scenarios that require continuous integration of new knowledge and skills. While continual learning (CL) offers a potential solution, existing benchmarks and methods suffer from critical limitations. In this paper, we introduce MLLM-CL, a novel benchmark encompassing domain and ability continual learning, where the former focuses on independently and identically distributed (IID) evaluation across evolving mainstream domains, whereas the latter evaluates on non-IID scenarios with emerging model ability. Methodologically, we propose preventing catastrophic interference through parameter isolation, along with an MLLM-based routing mechanism. Extensive experiments demonstrate that our approach can integrate domain-specific knowledge and functional abilities with minimal forgetting, significantly outperforming existing methods. 

---
# IntentionESC: An Intention-Centered Framework for Enhancing Emotional Support in Dialogue Systems 

**Authors**: Xinjie Zhang, Wenxuan Wang, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.05947)  

**Abstract**: In emotional support conversations, unclear intentions can lead supporters to employ inappropriate strategies, inadvertently imposing their expectations or solutions on the seeker. Clearly defined intentions are essential for guiding both the supporter's motivations and the overall emotional support process. In this paper, we propose the Intention-centered Emotional Support Conversation (IntentionESC) framework, which defines the possible intentions of supporters in emotional support conversations, identifies key emotional state aspects for inferring these intentions, and maps them to appropriate support strategies. While Large Language Models (LLMs) excel in text generating, they fundamentally operate as probabilistic models trained on extensive datasets, lacking a true understanding of human thought processes and intentions. To address this limitation, we introduce the Intention Centric Chain-of-Thought (ICECoT) mechanism. ICECoT enables LLMs to mimic human reasoning by analyzing emotional states, inferring intentions, and selecting suitable support strategies, thereby generating more effective emotional support responses. To train the model with ICECoT and integrate expert knowledge, we design an automated annotation pipeline that produces high-quality training data. Furthermore, we develop a comprehensive evaluation scheme to assess emotional support efficacy and conduct extensive experiments to validate our framework. Our data and code are available at this https URL. 

---
# Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry for Long-Context LLMs 

**Authors**: Wanyun Cui, Mingwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05410)  

**Abstract**: Recent advances in Large Language Models (LLMs) have highlighted the critical importance of extending context length, yet the quadratic complexity of attention mechanisms poses significant challenges for efficient long-context modeling. KV cache compression has emerged as a key approach to address this challenge. Through extensive empirical analysis, we reveal a fundamental yet previously overlooked asymmetry in KV caches: while adjacent keys receive similar attention weights (local homogeneity), adjacent values demonstrate distinct heterogeneous distributions. This key-value asymmetry reveals a critical limitation in existing compression methods that treat keys and values uniformly. To address the limitation, we propose a training-free compression framework (AsymKV) that combines homogeneity-based key merging with a mathematically proven lossless value compression. Extensive experiments demonstrate that AsymKV consistently outperforms existing long-context methods across various tasks and base models. For example, on LLaMA3.1-8B, AsymKV achieves an average score of 43.95 on LongBench, surpassing SOTA methods like H$_2$O (38.89) by a large margin. 

---
# Are Large Language Models Good Temporal Graph Learners? 

**Authors**: Shenyang Huang, Ali Parviz, Emma Kondrup, Zachary Yang, Zifeng Ding, Michael Bronstein, Reihaneh Rabbany, Guillaume Rabusseau  

**Link**: [PDF](https://arxiv.org/pdf/2506.05393)  

**Abstract**: Large Language Models (LLMs) have recently driven significant advancements in Natural Language Processing and various other applications. While a broad range of literature has explored the graph-reasoning capabilities of LLMs, including their use of predictors on graphs, the application of LLMs to dynamic graphs -- real world evolving networks -- remains relatively unexplored. Recent work studies synthetic temporal graphs generated by random graph models, but applying LLMs to real-world temporal graphs remains an open question. To address this gap, we introduce Temporal Graph Talker (TGTalker), a novel temporal graph learning framework designed for LLMs. TGTalker utilizes the recency bias in temporal graphs to extract relevant structural information, converted to natural language for LLMs, while leveraging temporal neighbors as additional information for prediction. TGTalker demonstrates competitive link prediction capabilities compared to existing Temporal Graph Neural Network (TGNN) models. Across five real-world networks, TGTalker performs competitively with state-of-the-art temporal graph methods while consistently outperforming popular models such as TGN and HTGN. Furthermore, TGTalker generates textual explanations for each prediction, thus opening up exciting new directions in explainability and interpretability for temporal link prediction. The code is publicly available at this https URL. 

---
# Advancing Decoding Strategies: Enhancements in Locally Typical Sampling for LLMs 

**Authors**: Jaydip Sen, Saptarshi Sengupta. Subhasis Dasgupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.05387)  

**Abstract**: This chapter explores advancements in decoding strategies for large language models (LLMs), focusing on enhancing the Locally Typical Sampling (LTS) algorithm. Traditional decoding methods, such as top-k and nucleus sampling, often struggle to balance fluency, diversity, and coherence in text generation. To address these challenges, Adaptive Semantic-Aware Typicality Sampling (ASTS) is proposed as an improved version of LTS, incorporating dynamic entropy thresholding, multi-objective scoring, and reward-penalty adjustments. ASTS ensures contextually coherent and diverse text generation while maintaining computational efficiency. Its performance is evaluated across multiple benchmarks, including story generation and abstractive summarization, using metrics such as perplexity, MAUVE, and diversity scores. Experimental results demonstrate that ASTS outperforms existing sampling techniques by reducing repetition, enhancing semantic alignment, and improving fluency. 

---
# Being Strong Progressively! Enhancing Knowledge Distillation of Large Language Models through a Curriculum Learning Framework 

**Authors**: Lingyuan Liu, Mengxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05695)  

**Abstract**: Knowledge Distillation (KD) compresses large language models (LLMs) by transferring the teacher model's capabilities to a smaller student model, reducing inference cost and memory usage while maintaining performance. However, existing KD methods for LLMs often fail to prevent significant shifts in the student model's distribution during training, leading to issues such as catastrophic forgetting, mode collapse, and training-inference mismatch. To address these challenges, we propose a novel, plug-in curriculum learning framework inspired by the strength training principle of "progressive overload" (POCL), which can be seamlessly integrated into existing white-box KD approaches with minimal computational overhead. The framework comprises two core components: (1) a difficulty measurer that ranks and partitions training samples from easy to hard, and (2) a training scheduler that incrementally introduces these subsets into the distillation process at fixed intervals while applying loss functions with progressively rising temperatures. By starting with the easiest samples and progressively increasing the difficulty, the approach enhances both the stability and efficiency of learning. Extensive experiments in instruction-following settings demonstrate that POCL consistently improves the performance of distilled student models across various white-box KD methods and model families. Our findings highlight the effectiveness of sorted training samples in KD for LLMs. More generally, our work demonstrates how to structure training data within the KD process to enhance the stability and performance of distilled LLMs. 

---
# LLMs Can Also Do Well! Breaking Barriers in Semantic Role Labeling via Large Language Models 

**Authors**: Xinxin Li, Huiyao Chen, Chengjun Liu, Jing Li, Meishan Zhang, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05385)  

**Abstract**: Semantic role labeling (SRL) is a crucial task of natural language processing (NLP). Although generative decoder-based large language models (LLMs) have achieved remarkable success across various NLP tasks, they still lag behind state-of-the-art encoder-decoder (BERT-like) models in SRL. In this work, we seek to bridge this gap by equipping LLMs for SRL with two mechanisms: (a) retrieval-augmented generation and (b) self-correction. The first mechanism enables LLMs to leverage external linguistic knowledge such as predicate and argument structure descriptions, while the second allows LLMs to identify and correct inconsistent SRL outputs. We conduct extensive experiments on three widely-used benchmarks of SRL (CPB1.0, CoNLL-2009, and CoNLL-2012). Results demonstrate that our method achieves state-of-the-art performance in both Chinese and English, marking the first successful application of LLMs to surpass encoder-decoder approaches in SRL. 

---
# Auto Review: Second Stage Error Detection for Highly Accurate Information Extraction from Phone Conversations 

**Authors**: Ayesha Qamar, Arushi Raghuvanshi, Conal Sathi, Youngseo Son  

**Link**: [PDF](https://arxiv.org/pdf/2506.05400)  

**Abstract**: Automating benefit verification phone calls saves time in healthcare and helps patients receive treatment faster. It is critical to obtain highly accurate information in these phone calls, as it can affect a patient's healthcare journey. Given the noise in phone call transcripts, we have a two-stage system that involves a post-call review phase for potentially noisy fields, where human reviewers manually verify the extracted data$\unicode{x2013}$a labor-intensive task. To automate this stage, we introduce Auto Review, which significantly reduces manual effort while maintaining a high bar for accuracy. This system, being highly reliant on call transcripts, suffers a performance bottleneck due to automatic speech recognition (ASR) issues. This problem is further exacerbated by the use of domain-specific jargon in the calls. In this work, we propose a second-stage postprocessing pipeline for accurate information extraction. We improve accuracy by using multiple ASR alternatives and a pseudo-labeling approach that does not require manually corrected transcripts. Experiments with general-purpose large language models and feature-based model pipelines demonstrate substantial improvements in the quality of corrected call transcripts, thereby enhancing the efficiency of Auto Review. 

---
# When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation 

**Authors**: Zhishang Xiang, Chuanjie Wu, Qinggang Zhang, Shengyuan Chen, Zijin Hong, Xiao Huang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.05690)  

**Abstract**: Graph retrieval-augmented generation (GraphRAG) has emerged as a powerful paradigm for enhancing large language models (LLMs) with external knowledge. It leverages graphs to model the hierarchical structure between specific concepts, enabling more coherent and effective knowledge retrieval for accurate this http URL its conceptual promise, recent studies report that GraphRAG frequently underperforms vanilla RAG on many real-world tasks. This raises a critical question: Is GraphRAG really effective, and in which scenarios do graph structures provide measurable benefits for RAG systems? To address this, we propose GraphRAG-Bench, a comprehensive benchmark designed to evaluate GraphRAG models onboth hierarchical knowledge retrieval and deep contextual reasoning. GraphRAG-Bench features a comprehensive dataset with tasks of increasing difficulty, coveringfact retrieval, complex reasoning, contextual summarization, and creative generation, and a systematic evaluation across the entire pipeline, from graph constructionand knowledge retrieval to final generation. Leveraging this novel benchmark, we systematically investigate the conditions when GraphRAG surpasses traditional RAG and the underlying reasons for its success, offering guidelines for its practical application. All related resources and analyses are collected for the community at this https URL. 

---
# PersonaAgent: When Large Language Model Agents Meet Personalization at Test Time 

**Authors**: Weizhi Zhang, Xinyang Zhang, Chenwei Zhang, Liangwei Yang, Jingbo Shang, Zhepei Wei, Henry Peng Zou, Zijie Huang, Zhengyang Wang, Yifan Gao, Xiaoman Pan, Lian Xiong, Jingguo Liu, Philip S. Yu, Xian Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06254)  

**Abstract**: Large Language Model (LLM) empowered agents have recently emerged as advanced paradigms that exhibit impressive capabilities in a wide range of domains and tasks. Despite their potential, current LLM agents often adopt a one-size-fits-all approach, lacking the flexibility to respond to users' varying needs and preferences. This limitation motivates us to develop PersonaAgent, the first personalized LLM agent framework designed to address versatile personalization tasks. Specifically, PersonaAgent integrates two complementary components - a personalized memory module that includes episodic and semantic memory mechanisms; a personalized action module that enables the agent to perform tool actions tailored to the user. At the core, the persona (defined as unique system prompt for each user) functions as an intermediary: it leverages insights from personalized memory to control agent actions, while the outcomes of these actions in turn refine the memory. Based on the framework, we propose a test-time user-preference alignment strategy that simulate the latest n interactions to optimize the persona prompt, ensuring real-time user preference alignment through textual loss feedback between simulated and ground-truth responses. Experimental evaluations demonstrate that PersonaAgent significantly outperforms other baseline methods by not only personalizing the action space effectively but also scaling during test-time real-world applications. These results underscore the feasibility and potential of our approach in delivering tailored, dynamic user experiences. 

---
# Improving LLMs with a knowledge from databases 

**Authors**: Petr Máša  

**Link**: [PDF](https://arxiv.org/pdf/2506.05560)  

**Abstract**: Large language models (LLMs) are achieving significant progress almost every moment now. Many advanced techniques have been introduced and widely accepted, like retrieval-augmentation generation (RAG), agents, and tools. Tools can query the database to answer questions from structured data files or perform groupings or other statistics. This unlocks huge opportunities, such as it can answer any question, but also poses threats, such as safety, because there is no control over the commands that are created. We would like to discuss whether we can create a new method that improves answers based on dataset/database via some interpretable ML methods, namely enhanced association rules. The advantage would be if the method can be also used in some safe technique like RAG. Association rules have a sound history. Since the introduction of CN2 and aproiri, many enhancements have been made. In parallel, enhanced association rules have been introduced and evolved over the last 40 years. The general problem is typically that there are too many rules. There are some techniques for handling it, but when LLM emerged, it turned out to be the best use case for the RAG technique for LLMs. We proposed a method that generates a ruleset based on defined knowledge patterns, then converts rules into text form via a rule-to-text converter, and includes the result as an RAG into LLM. We compared this method with ChatGPT (even with using agents) and we have discovered a significant improvement in answering questions based on the dataset. We have also tried several strategies how much rules to generate. We found this improvement interesting. Moreover, it can also be improved in many ways as future work, like incorporating other patterns, the use of rule mining as an agent, and many others. 

---
# The Lock-in Hypothesis: Stagnation by Algorithm 

**Authors**: Tianyi Alex Qiu, Zhonghao He, Tejasveer Chugh, Max Kleiman-Weiner  

**Link**: [PDF](https://arxiv.org/pdf/2506.06166)  

**Abstract**: The training and deployment of large language models (LLMs) create a feedback loop with human users: models learn human beliefs from data, reinforce these beliefs with generated content, reabsorb the reinforced beliefs, and feed them back to users again and again. This dynamic resembles an echo chamber. We hypothesize that this feedback loop entrenches the existing values and beliefs of users, leading to a loss of diversity and potentially the lock-in of false beliefs. We formalize this hypothesis and test it empirically with agent-based LLM simulations and real-world GPT usage data. Analysis reveals sudden but sustained drops in diversity after the release of new GPT iterations, consistent with the hypothesized human-AI feedback loop. Code and data available at this https URL 

---
# Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs 

**Authors**: Ananth Muppidi, Abhilash Nandy, Sambaran Bandyopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2506.05629)  

**Abstract**: The performance of large language models in domain-specific tasks necessitates fine-tuning, which is computationally expensive and technically challenging. This paper focuses on parameter-efficient fine-tuning using soft prompting, a promising approach that adapts pre-trained models to downstream tasks by learning a small set of parameters. We propose a novel Input Dependent Soft Prompting technique with a self-Attention Mechanism (ID-SPAM) that generates soft prompts based on the input tokens and attends different tokens with varying importance. Our method is simple and efficient, keeping the number of trainable parameters small. We show the merits of the proposed approach compared to state-of-the-art techniques on various tasks and show the improved zero shot domain transfer capability. 

---
# Audio-Aware Large Language Models as Judges for Speaking Styles 

**Authors**: Cheng-Han Chiang, Xiaofei Wang, Chung-Ching Lin, Kevin Lin, Linjie Li, Radu Kopetz, Yao Qian, Zhendong Wang, Zhengyuan Yang, Hung-yi Lee, Lijuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05984)  

**Abstract**: Audio-aware large language models (ALLMs) can understand the textual and non-textual information in the audio input. In this paper, we explore using ALLMs as an automatic judge to assess the speaking styles of speeches. We use ALLM judges to evaluate the speeches generated by SLMs on two tasks: voice style instruction following and role-playing. The speaking style we consider includes emotion, volume, speaking pace, word emphasis, pitch control, and non-verbal elements. We use four spoken language models (SLMs) to complete the two tasks and use humans and ALLMs to judge the SLMs' responses. We compare two ALLM judges, GPT-4o-audio and Gemini-2.5-pro, with human evaluation results and show that the agreement between Gemini and human judges is comparable to the agreement between human evaluators. These promising results show that ALLMs can be used as a judge to evaluate SLMs. Our results also reveal that current SLMs, even GPT-4o-audio, still have room for improvement in controlling the speaking style and generating natural dialogues. 

---
# Corrector Sampling in Language Models 

**Authors**: Itai Gat, Neta Shaul, Uriel Singer, Yaron Lipman  

**Link**: [PDF](https://arxiv.org/pdf/2506.06215)  

**Abstract**: Autoregressive language models accumulate errors due to their fixed, irrevocable left-to-right token generation. To address this, we propose a new sampling method called Resample-Previous-Tokens (RPT). RPT mitigates error accumulation by iteratively revisiting and potentially replacing tokens in a window of previously generated text. This method can be integrated into existing autoregressive models, preserving their next-token-prediction quality and speed. Fine-tuning a pretrained 8B parameter model with RPT for only 100B resulted in ~10% relative improvements on reasoning and coding benchmarks compared to the standard sampling. 

---
# Do Large Vision-Language Models Distinguish between the Actual and Apparent Features of Illusions? 

**Authors**: Taiga Shinozaki, Tomoki Doi, Satoshi Nishida, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2506.05765)  

**Abstract**: Humans are susceptible to optical illusions, which serve as valuable tools for investigating sensory and cognitive processes. Inspired by human vision studies, research has begun exploring whether machines, such as large vision language models (LVLMs), exhibit similar susceptibilities to visual illusions. However, studies often have used non-abstract images and have not distinguished actual and apparent features, leading to ambiguous assessments of machine cognition. To address these limitations, we introduce a visual question answering (VQA) dataset, categorized into genuine and fake illusions, along with corresponding control images. Genuine illusions present discrepancies between actual and apparent features, whereas fake illusions have the same actual and apparent features even though they look illusory due to the similar geometric configuration. We evaluate the performance of LVLMs for genuine and fake illusion VQA tasks and investigate whether the models discern actual and apparent features. Our findings indicate that although LVLMs may appear to recognize illusions by correctly answering questions about both feature types, they predict the same answers for both Genuine Illusion and Fake Illusion VQA questions. This suggests that their responses might be based on prior knowledge of illusions rather than genuine visual understanding. The dataset is available at this https URL 

---
# Beyond RAG: Reinforced Reasoning Augmented Generation for Clinical Notes 

**Authors**: Lo Pang-Yun Ting, Chengshuai Zhao, Yu-Hua Zeng, Yuan Jee Lim, Kun-Ta Chuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05386)  

**Abstract**: Clinical note generation aims to automatically produce free-text summaries of a patient's condition and diagnostic process, with discharge instructions being a representative long-form example. While recent large language model (LLM)-based methods pre-trained on general clinical corpora show promise in clinical text generation, they fall short in producing long-form notes from limited patient information. In this paper, we propose R2AG, the first reinforced retriever for long-form discharge instruction generation based on pre-admission data. R2AG is trained with reinforcement learning to retrieve reasoning paths from a medical knowledge graph, providing explicit semantic guidance to the LLM. To bridge the information gap, we propose Group-Based Retriever Optimization (GRO) which improves retrieval quality with group-relative rewards, encouraging reasoning leaps for deeper inference by the LLM. Comprehensive experiments on the MIMIC-IV-Note dataset show that R2AG outperforms baselines in both clinical efficacy and natural language generation metrics. Further analysis reveals that R2AG fills semantic gaps in sparse input scenarios, and retrieved reasoning paths help LLMs avoid clinical misinterpretation by focusing on key evidence and following coherent reasoning. 

---
# CodeContests+: High-Quality Test Case Generation for Competitive Programming 

**Authors**: Zihan Wang, Siyao Liu, Yang Sun, Hongyan Li, Kai Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.05817)  

**Abstract**: Competitive programming, due to its high reasoning difficulty and precise correctness feedback, has become a key task for both training and evaluating the reasoning capabilities of large language models (LLMs). However, while a large amount of public problem data, such as problem statements and solutions, is available, the test cases of these problems are often difficult to obtain. Therefore, test case generation is a necessary task for building large-scale datasets, and the quality of the test cases directly determines the accuracy of the evaluation. In this paper, we introduce an LLM-based agent system that creates high-quality test cases for competitive programming problems. We apply this system to the CodeContests dataset and propose a new version with improved test cases, named CodeContests+. We evaluated the quality of test cases in CodeContestsPlus. First, we used 1.72 million submissions with pass/fail labels to examine the accuracy of these test cases in evaluation. The results indicated that CodeContests+ achieves significantly higher accuracy than CodeContests, particularly with a notably higher True Positive Rate (TPR). Subsequently, our experiments in LLM Reinforcement Learning (RL) further confirmed that improvements in test case quality yield considerable advantages for RL. 

---
# Contextually Guided Transformers via Low-Rank Adaptation 

**Authors**: Andrey Zhmoginov, Jihwan Lee, Max Vladymyrov, Mark Sandler  

**Link**: [PDF](https://arxiv.org/pdf/2506.05672)  

**Abstract**: Large Language Models (LLMs) based on Transformers excel at text processing, but their reliance on prompts for specialized behavior introduces computational overhead. We propose a modification to a Transformer architecture that eliminates the need for explicit prompts by learning to encode context into the model's weights. Our Contextually Guided Transformer (CGT) model maintains a contextual summary at each sequence position, allowing it to update the weights on the fly based on the preceding context. This approach enables the model to self-specialize, effectively creating a tailored model for processing information following a given prefix. We demonstrate the effectiveness of our method on synthetic in-context learning tasks and language modeling benchmarks. Furthermore, we introduce techniques for enhancing the interpretability of the learned contextual representations, drawing connections to Variational Autoencoders and promoting smoother, more consistent context encoding. This work offers a novel direction for efficient and adaptable language modeling by integrating context directly into the model's architecture. 

---
# Masked Language Models are Good Heterogeneous Graph Generalizers 

**Authors**: Jinyu Yang, Cheng Yang, Shanyuan Cui, Zeyuan Guo, Liangwei Yang, Muhan Zhang, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06157)  

**Abstract**: Heterogeneous graph neural networks (HGNNs) excel at capturing structural and semantic information in heterogeneous graphs (HGs), while struggling to generalize across domains and tasks. Recently, some researchers have turned to integrating HGNNs with large language models (LLMs) for more generalizable heterogeneous graph learning. However, these approaches typically extract structural information via HGNNs as HG tokens, and disparities in embedding spaces between HGNNs and LLMs have been shown to bias the LLM's comprehension of HGs. Moreover, as these HG tokens are often derived from node-level tasks, the model's ability to generalize across tasks remains limited. To this end, we propose a simple yet effective Masked Language Modeling-based method, called MLM4HG. MLM4HG introduces metapath-based textual sequences instead of HG tokens to extract structural and semantic information inherent in HGs, and designs customized textual templates to unify different graph tasks into a coherent cloze-style "mask" token prediction paradigm. Specifically, MLM4HG first converts HGs from various domains to texts based on metapaths, and subsequently combines them with the unified task texts to form a HG-based corpus. Moreover, the corpus is fed into a pretrained LM for fine-tuning with a constrained target vocabulary, enabling the fine-tuned LM to generalize to unseen target HGs. Extensive cross-domain and multi-task experiments on four real-world datasets demonstrate the superior generalization performance of MLM4HG over state-of-the-art methods in both few-shot and zero-shot scenarios. Our code is available at this https URL. 

---
# Deployability-Centric Infrastructure-as-Code Generation: An LLM-based Iterative Framework 

**Authors**: Tianyi Zhang, Shidong Pan, Zejun Zhang, Zhenchang Xing, Xiaoyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.05623)  

**Abstract**: Infrastructure-as-Code (IaC) generation holds significant promise for automating cloud infrastructure provisioning. Recent advances in Large Language Models (LLMs) present a promising opportunity to democratize IaC development by generating deployable infrastructure templates from natural language descriptions, but current evaluation focuses on syntactic correctness while ignoring deployability, the fatal measure of IaC template utility. We address this gap through two contributions: (1) IaCGen, an LLM-based deployability-centric framework that uses iterative feedback mechanism to generate IaC templates, and (2) DPIaC-Eval, a deployability-centric IaC template benchmark consists of 153 real-world scenarios that can evaluate syntax, deployment, user intent, and security. Our evaluation reveals that state-of-the-art LLMs initially performed poorly, with Claude-3.5 and Claude-3.7 achieving only 30.2% and 26.8% deployment success on the first attempt respectively. However, IaCGen transforms this performance dramatically: all evaluated models reach over 90% passItr@25, with Claude-3.5 and Claude-3.7 achieving 98% success rate. Despite these improvements, critical challenges remain in user intent alignment (25.2% accuracy) and security compliance (8.4% pass rate), highlighting areas requiring continued research. Our work provides the first comprehensive assessment of deployability-centric IaC template generation and establishes a foundation for future research. 

---
# Voice Impression Control in Zero-Shot TTS 

**Authors**: Keinichi Fujita, Shota Horiguchi, Yusuke Ijima  

**Link**: [PDF](https://arxiv.org/pdf/2506.05688)  

**Abstract**: Para-/non-linguistic information in speech is pivotal in shaping the listeners' impression. Although zero-shot text-to-speech (TTS) has achieved high speaker fidelity, modulating subtle para-/non-linguistic information to control perceived voice characteristics, i.e., impressions, remains challenging. We have therefore developed a voice impression control method in zero-shot TTS that utilizes a low-dimensional vector to represent the intensities of various voice impression pairs (e.g., dark-bright). The results of both objective and subjective evaluations have demonstrated our method's effectiveness in impression control. Furthermore, generating this vector via a large language model enables target-impression generation from a natural language description of the desired impression, thus eliminating the need for manual optimization. 

---
# Interpretation Meets Safety: A Survey on Interpretation Methods and Tools for Improving LLM Safety 

**Authors**: Seongmin Lee, Aeree Cho, Grace C. Kim, ShengYun Peng, Mansi Phute, Duen Horng Chau  

**Link**: [PDF](https://arxiv.org/pdf/2506.05451)  

**Abstract**: As large language models (LLMs) see wider real-world use, understanding and mitigating their unsafe behaviors is critical. Interpretation techniques can reveal causes of unsafe outputs and guide safety, but such connections with safety are often overlooked in prior surveys. We present the first survey that bridges this gap, introducing a unified framework that connects safety-focused interpretation methods, the safety enhancements they inform, and the tools that operationalize them. Our novel taxonomy, organized by LLM workflow stages, summarizes nearly 70 works at their intersections. We conclude with open challenges and future directions. This timely survey helps researchers and practitioners navigate key advancements for safer, more interpretable LLMs. 

---
# Low-Resource Domain Adaptation for Speech LLMs via Text-Only Fine-Tuning 

**Authors**: Yangui Fang, Jing Peng, Xu Li, Yu Xi, Chengwei Zhang, Guohui Zhong, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05671)  

**Abstract**: Recent advances in automatic speech recognition (ASR) have combined speech encoders with large language models (LLMs) through projection, forming Speech LLMs with strong performance. However, adapting them to new domains remains challenging, especially in low-resource settings where paired speech-text data is scarce. We propose a text-only fine-tuning strategy for Speech LLMs using unpaired target-domain text without requiring additional audio. To preserve speech-text alignment, we introduce a real-time evaluation mechanism during fine-tuning. This enables effective domain adaptation while maintaining source-domain performance. Experiments on LibriSpeech, SlideSpeech, and Medical datasets show that our method achieves competitive recognition performance, with minimal degradation compared to full audio-text fine-tuning. It also improves generalization to new domains without catastrophic forgetting, highlighting the potential of text-only fine-tuning for low-resource domain adaptation of ASR. 

---
# Research on Personalized Financial Product Recommendation by Integrating Large Language Models and Graph Neural Networks 

**Authors**: Yushang Zhao, Yike Peng, Dannier Li, Yuxin Yang, Chengrui Zhou, Jing Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.05873)  

**Abstract**: With the rapid growth of fintech, personalized financial product recommendations have become increasingly important. Traditional methods like collaborative filtering or content-based models often fail to capture users' latent preferences and complex relationships. We propose a hybrid framework integrating large language models (LLMs) and graph neural networks (GNNs). A pre-trained LLM encodes text data (e.g., user reviews) into rich feature vectors, while a heterogeneous user-product graph models interactions and social ties. Through a tailored message-passing mechanism, text and graph information are fused within the GNN to jointly optimize embeddings. Experiments on public and real-world financial datasets show our model outperforms standalone LLM or GNN in accuracy, recall, and NDCG, with strong interpretability. This work offers new insights for personalized financial recommendations and cross-modal fusion in broader recommendation tasks. 

---
# On the Merits of LLM-Based Corpus Enrichment 

**Authors**: Gal Zur, Tommy Mordo, Moshe Tennenholtz, Oren Kurland  

**Link**: [PDF](https://arxiv.org/pdf/2506.06015)  

**Abstract**: Generative AI (genAI) technologies -- specifically, large language models (LLMs) -- and search have evolving relations. We argue for a novel perspective: using genAI to enrich a document corpus so as to improve query-based retrieval effectiveness. The enrichment is based on modifying existing documents or generating new ones. As an empirical proof of concept, we use LLMs to generate documents relevant to a topic which are more retrievable than existing ones. In addition, we demonstrate the potential merits of using corpus enrichment for retrieval augmented generation (RAG) and answer attribution in question answering. 

---
# CP-Bench: Evaluating Large Language Models for Constraint Modelling 

**Authors**: Kostis Michailidis, Dimos Tsouros, Tias Guns  

**Link**: [PDF](https://arxiv.org/pdf/2506.06052)  

**Abstract**: Combinatorial problems are present in a wide range of industries. Constraint Programming (CP) is a well-suited problem-solving paradigm, but its core process, namely constraint modelling, is a bottleneck for wider adoption. Aiming to alleviate this bottleneck, recent studies have explored using Large Language Models (LLMs) as modelling assistants, transforming combinatorial problem descriptions to executable constraint models, similar to coding assistants. However, the existing evaluation datasets for constraint modelling are often limited to small, homogeneous, or domain-specific instances, which do not capture the diversity of real-world scenarios. This work addresses this gap by introducing CP-Bench, a novel benchmark dataset that includes a diverse set of well-known combinatorial problem classes sourced from the CP community, structured explicitly for evaluating LLM-driven CP modelling. With this dataset, and given the variety of constraint modelling frameworks, we compare and evaluate the modelling capabilities of LLMs for three distinct constraint modelling systems, which vary in abstraction level and underlying syntax: the high-level MiniZinc language and Python-based CPMpy library, and the lower-level Python interface of the OR-Tools CP-SAT solver. In order to enhance the ability of LLMs to produce valid constraint models, we systematically evaluate the use of prompt-based and inference-time compute methods adapted from existing LLM-based code generation research. Our results underscore the modelling convenience provided by Python-based frameworks, as well as the effectiveness of documentation-rich system prompts, which, augmented with repeated sampling and self-verification, achieve further improvements, reaching up to 70\% accuracy on this new, highly challenging benchmark. 

---
# CrimeMind: Simulating Urban Crime with Multi-Modal LLM Agents 

**Authors**: Qingbin Zeng, Ruotong Zhao, Jinzhu Mao, Haoyang Li, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05981)  

**Abstract**: Modeling urban crime is an important yet challenging task that requires understanding the subtle visual, social, and cultural cues embedded in urban environments. Previous work has predominantly focused on rule-based agent-based modeling (ABM) and deep learning methods. ABMs offer interpretability of internal mechanisms but exhibit limited predictive this http URL contrast, deep learning methods are often effective in prediction but are less interpretable and require extensive training data. Moreover, both lines of work lack the cognitive flexibility to adapt to changing environments. Leveraging the capabilities of large language models (LLMs), we propose CrimeMind, a novel LLM-driven ABM framework for simulating urban crime within a multi-modal urban context.A key innovation of our design is the integration of the Routine Activity Theory (RAT) into the agentic workflow of CrimeMind, enabling it to process rich multi-modal urban features and reason about criminal this http URL, RAT requires LLM agents to infer subtle cues in evaluating environmental safety as part of assessing guardianship, which can be challenging for LLMs. To address this, we collect a small-scale human-annotated dataset and align CrimeMind's perception with human judgment via a training-free textual gradient this http URL across four major U.S. cities demonstrate that CrimeMind outperforms both traditional ABMs and deep learning baselines in crime hotspot prediction and spatial distribution accuracy, achieving up to a 24% improvement over the strongest this http URL, we conduct counterfactual simulations of external incidents and policy interventions and it successfully captures the expected changes in crime patterns, demonstrating its ability to reflect counterfactual this http URL, CrimeMind enables fine-grained modeling of individual behaviors and facilitates evaluation of real-world interventions. 

---
# Toward Greater Autonomy in Materials Discovery Agents: Unifying Planning, Physics, and Scientists 

**Authors**: Lianhao Zhou, Hongyi Ling, Keqiang Yan, Kaiji Zhao, Xiaoning Qian, Raymundo Arróyave, Xiaofeng Qian, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2506.05616)  

**Abstract**: We aim at designing language agents with greater autonomy for crystal materials discovery. While most of existing studies restrict the agents to perform specific tasks within predefined workflows, we aim to automate workflow planning given high-level goals and scientist intuition. To this end, we propose Materials Agent unifying Planning, Physics, and Scientists, known as MAPPS. MAPPS consists of a Workflow Planner, a Tool Code Generator, and a Scientific Mediator. The Workflow Planner uses large language models (LLMs) to generate structured and multi-step workflows. The Tool Code Generator synthesizes executable Python code for various tasks, including invoking a force field foundation model that encodes physics. The Scientific Mediator coordinates communications, facilitates scientist feedback, and ensures robustness through error reflection and recovery. By unifying planning, physics, and scientists, MAPPS enables flexible and reliable materials discovery with greater autonomy, achieving a five-fold improvement in stability, uniqueness, and novelty rates compared with prior generative models when evaluated on the MP-20 data. We provide extensive experiments across diverse tasks to show that MAPPS is a promising framework for autonomous materials discovery. 

---
# Explainability in Context: A Multilevel Framework Aligning AI Explanations with Stakeholder with LLMs 

**Authors**: Marilyn Bello, Rafael Bello, Maria-Matilde García, Ann Nowé, Iván Sevillano-García, Francisco Herrera  

**Link**: [PDF](https://arxiv.org/pdf/2506.05887)  

**Abstract**: The growing application of artificial intelligence in sensitive domains has intensified the demand for systems that are not only accurate but also explainable and trustworthy. Although explainable AI (XAI) methods have proliferated, many do not consider the diverse audiences that interact with AI systems: from developers and domain experts to end-users and society. This paper addresses how trust in AI is influenced by the design and delivery of explanations and proposes a multilevel framework that aligns explanations with the epistemic, contextual, and ethical expectations of different stakeholders. The framework consists of three layers: algorithmic and domain-based, human-centered, and social explainability. We highlight the emerging role of Large Language Models (LLMs) in enhancing the social layer by generating accessible, natural language explanations. Through illustrative case studies, we demonstrate how this approach facilitates technical fidelity, user engagement, and societal accountability, reframing XAI as a dynamic, trust-building process. 

---
# Cartridges: Lightweight and general-purpose long context representations via self-study 

**Authors**: Sabri Eyuboglu, Ryan Ehrlich, Simran Arora, Neel Guha, Dylan Zinsley, Emily Liu, Will Tennien, Atri Rudra, James Zou, Azalia Mirhoseini, Christopher Re  

**Link**: [PDF](https://arxiv.org/pdf/2506.06266)  

**Abstract**: Large language models are often used to answer queries grounded in large text corpora (e.g. codebases, legal documents, or chat histories) by placing the entire corpus in the context window and leveraging in-context learning (ICL). Although current models support contexts of 100K-1M tokens, this setup is costly to serve because the memory consumption of the KV cache scales with input length. We explore an alternative: training a smaller KV cache offline on each corpus. At inference time, we load this trained KV cache, which we call a Cartridge, and decode a response. Critically, the cost of training a Cartridge can be amortized across all the queries referencing the same corpus. However, we find that the naive approach of training the Cartridge with next-token prediction on the corpus is not competitive with ICL. Instead, we propose self-study, a training recipe in which we generate synthetic conversations about the corpus and train the Cartridge with a context-distillation objective. We find that Cartridges trained with self-study replicate the functionality of ICL, while being significantly cheaper to serve. On challenging long-context benchmarks, Cartridges trained with self-study match ICL performance while using 38.6x less memory and enabling 26.4x higher throughput. Self-study also extends the model's effective context length (e.g. from 128k to 484k tokens on MTOB) and surprisingly, leads to Cartridges that can be composed at inference time without retraining. 

---
# Eigenspectrum Analysis of Neural Networks without Aspect Ratio Bias 

**Authors**: Yuanzhe Hu, Kinshuk Goel, Vlad Killiakov, Yaoqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06280)  

**Abstract**: Diagnosing deep neural networks (DNNs) through the eigenspectrum of weight matrices has been an active area of research in recent years. At a high level, eigenspectrum analysis of DNNs involves measuring the heavytailness of the empirical spectral densities (ESD) of weight matrices. It provides insight into how well a model is trained and can guide decisions on assigning better layer-wise training hyperparameters. In this paper, we address a challenge associated with such eigenspectrum methods: the impact of the aspect ratio of weight matrices on estimated heavytailness metrics. We demonstrate that matrices of varying sizes (and aspect ratios) introduce a non-negligible bias in estimating heavytailness metrics, leading to inaccurate model diagnosis and layer-wise hyperparameter assignment. To overcome this challenge, we propose FARMS (Fixed-Aspect-Ratio Matrix Subsampling), a method that normalizes the weight matrices by subsampling submatrices with a fixed aspect ratio. Instead of measuring the heavytailness of the original ESD, we measure the average ESD of these subsampled submatrices. We show that measuring the heavytailness of these submatrices with the fixed aspect ratio can effectively mitigate the aspect ratio bias. We validate our approach across various optimization techniques and application domains that involve eigenspectrum analysis of weights, including image classification in computer vision (CV) models, scientific machine learning (SciML) model training, and large language model (LLM) pruning. Our results show that despite its simplicity, FARMS uniformly improves the accuracy of eigenspectrum analysis while enabling more effective layer-wise hyperparameter assignment in these application domains. In one of the LLM pruning experiments, FARMS reduces the perplexity of the LLaMA-7B model by 17.3% when compared with the state-of-the-art method. 

---
# Topology of Reasoning: Understanding Large Reasoning Models through Reasoning Graph Properties 

**Authors**: Gouki Minegishi, Hiroki Furuta, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2506.05744)  

**Abstract**: Recent large-scale reasoning models have achieved state-of-the-art performance on challenging mathematical benchmarks, yet the internal mechanisms underlying their success remain poorly understood. In this work, we introduce the notion of a reasoning graph, extracted by clustering hidden-state representations at each reasoning step, and systematically analyze three key graph-theoretic properties: cyclicity, diameter, and small-world index, across multiple tasks (GSM8K, MATH500, AIME 2024). Our findings reveal that distilled reasoning models (e.g., DeepSeek-R1-Distill-Qwen-32B) exhibit significantly more recurrent cycles (about 5 per sample), substantially larger graph diameters, and pronounced small-world characteristics (about 6x) compared to their base counterparts. Notably, these structural advantages grow with task difficulty and model capacity, with cycle detection peaking at the 14B scale and exploration diameter maximized in the 32B variant, correlating positively with accuracy. Furthermore, we show that supervised fine-tuning on an improved dataset systematically expands reasoning graph diameters in tandem with performance gains, offering concrete guidelines for dataset design aimed at boosting reasoning capabilities. By bridging theoretical insights into reasoning graph structures with practical recommendations for data construction, our work advances both the interpretability and the efficacy of large reasoning models. 

---
# Astra: Toward General-Purpose Mobile Robots via Hierarchical Multimodal Learning 

**Authors**: Sheng Chen, Peiyu He, Jiaxin Hu, Ziyang Liu, Yansheng Wang, Tao Xu, Chi Zhang, Chongchong Zhang, Chao An, Shiyu Cai, Duo Cao, Kangping Chen, Shuai Chu, Tianwei Chu, Mingdi Dan, Min Du, Weiwei Fang, Pengyou Fu, Junkai Hu, Xiaowei Jiang, Zhaodi Jiang, Fuxuan Li, Jun Li, Minghui Li, Mingyao Li, Yanchang Li, Zhibin Li, Guangming Liu, Kairui Liu, Lihao Liu, Weizhi Liu, Xiaoshun Liu, Yufei Liu, Yunfei Liu, Qiang Lu, Yuanfei Luo, Xiang Lv, Hongying Ma, Sai Ma, Lingxian Mi, Sha Sa, Hongxiang Shu, Lei Tian, Chengzhi Wang, Jiayu Wang, Kaijie Wang, Qingyi Wang, Renwen Wang, Tao Wang, Wei Wang, Xirui Wang, Chao Wei, Xuguang Wei, Zijun Xia, Zhaohao Xiao, Tingshuai Yan, Liyan Yang, Yifan Yang, Zhikai Yang, Zhong Yin, Li Yuan, Liuchun Yuan, Chi Zhang, Jinyang Zhang, Junhui Zhang, Linge Zhang, Zhenyi Zhang, Zheyu Zhang, Dongjie Zhu, Hang Li, Yangang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06205)  

**Abstract**: Modern robot navigation systems encounter difficulties in diverse and complex indoor environments. Traditional approaches rely on multiple modules with small models or rule-based systems and thus lack adaptability to new environments. To address this, we developed Astra, a comprehensive dual-model architecture, Astra-Global and Astra-Local, for mobile robot navigation. Astra-Global, a multimodal LLM, processes vision and language inputs to perform self and goal localization using a hybrid topological-semantic graph as the global map, and outperforms traditional visual place recognition methods. Astra-Local, a multitask network, handles local path planning and odometry estimation. Its 4D spatial-temporal encoder, trained through self-supervised learning, generates robust 4D features for downstream tasks. The planning head utilizes flow matching and a novel masked ESDF loss to minimize collision risks for generating local trajectories, and the odometry head integrates multi-sensor inputs via a transformer encoder to predict the relative pose of the robot. Deployed on real in-house mobile robots, Astra achieves high end-to-end mission success rate across diverse indoor environments. 

---
# Text-to-LoRA: Instant Transformer Adaption 

**Authors**: Rujikorn Charakorn, Edoardo Cetin, Yujin Tang, Robert Tjarko Lange  

**Link**: [PDF](https://arxiv.org/pdf/2506.06105)  

**Abstract**: While Foundation Models provide a general tool for rapid content creation, they regularly require task-specific adaptation. Traditionally, this exercise involves careful curation of datasets and repeated fine-tuning of the underlying model. Fine-tuning techniques enable practitioners to adapt foundation models for many new applications but require expensive and lengthy training while being notably sensitive to hyper-parameter choices. To overcome these limitations, we introduce Text-to-LoRA (T2L), a model capable of adapting Large Language Models on the fly solely based on a natural language description of the target task. T2L is a hypernetwork trained to construct LoRAs in a single inexpensive forward pass. After training T2L on a suite of 9 pre-trained LoRA adapters (GSM8K, Arc, etc.), we show that the ad-hoc reconstructed LoRA instances match the performance of task-specific adapters across the corresponding test sets. Furthermore, T2L can compress hundreds of LoRA instances and zero-shot generalize to entirely unseen tasks. This approach provides a significant step towards democratizing the specialization of foundation models and enables language-based adaptation with minimal compute requirements. Our code is available at this https URL 

---
# Leveraging Generative AI for Enhancing Automated Assessment in Programming Education Contests 

**Authors**: Stefan Dascalescu, Adrian Marius Dumitran, Mihai Alexandru Vasiluta  

**Link**: [PDF](https://arxiv.org/pdf/2506.05990)  

**Abstract**: Competitive programming contests play a crucial role in cultivating computational thinking and algorithmic skills among learners. However, generating comprehensive test cases to effectively assess programming solutions remains resource-intensive and challenging for educators. This paper introduces an innovative NLP-driven method leveraging generative AI (large language models) to automate the creation of high-quality test cases for competitive programming assessments. We extensively evaluated our approach on diverse datasets, including 25 years of Romanian Informatics Olympiad (OJI) data for 5th graders, recent competitions hosted on the this http URL platform, and the International Informatics Olympiad in Teams (IIOT). Our results demonstrate that AI-generated test cases substantially enhanced assessments, notably identifying previously undetected errors in 67% of the OJI 5th grade programming problems. These improvements underscore the complementary educational value of our technique in formative assessment contexts. By openly sharing our prompts, translated datasets, and methodologies, we offer practical NLP-based tools that educators and contest organizers can readily integrate to enhance assessment quality, reduce workload, and deepen insights into learner performance. 

---
# Small Models, Big Support: A Local LLM Framework for Teacher-Centric Content Creation and Assessment using RAG and CAG 

**Authors**: Zarreen Reza, Alexander Mazur, Michael T. Dugdale, Robin Ray-Chaudhuri  

**Link**: [PDF](https://arxiv.org/pdf/2506.05925)  

**Abstract**: While Large Language Models (LLMs) are increasingly utilized as student-facing educational aids, their potential to directly support educators, particularly through locally deployable and customizable open-source solutions, remains significantly underexplored. Many existing educational solutions rely on cloud-based infrastructure or proprietary tools, which are costly and may raise privacy concerns. Regulated industries with limited budgets require affordable, self-hosted solutions. We introduce an end-to-end, open-source framework leveraging small (3B-7B parameters), locally deployed LLMs for customized teaching material generation and assessment. Our system uniquely incorporates an interactive loop crucial for effective small-model refinement, and an auxiliary LLM verifier to mitigate jailbreaking risks, enhancing output reliability and safety. Utilizing Retrieval and Context Augmented Generation (RAG/CAG), it produces factually accurate, customized pedagogically-styled content. Deployed on-premises for data privacy and validated through an evaluation pipeline and a college physics pilot, our findings show that carefully engineered small LLM systems can offer robust, affordable, practical, and safe educator support, achieving utility comparable to larger models for targeted tasks. 

---
# Bayesian Persuasion as a Bargaining Game 

**Authors**: Yue Lin, Shuhui Zhu, William A Cunningham, Wenhao Li, Pascal Poupart, Hongyuan Zha, Baoxiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05876)  

**Abstract**: Bayesian persuasion, an extension of cheap-talk communication, involves an informed sender committing to a signaling scheme to influence a receiver's actions. Compared to cheap talk, this sender's commitment enables the receiver to verify the incentive compatibility of signals beforehand, facilitating cooperation. While effective in one-shot scenarios, Bayesian persuasion faces computational complexity (NP-hardness) when extended to long-term interactions, where the receiver may adopt dynamic strategies conditional on past outcomes and future expectations. To address this complexity, we introduce the bargaining perspective, which allows: (1) a unified framework and well-structured solution concept for long-term persuasion, with desirable properties such as fairness and Pareto efficiency; (2) a clear distinction between two previously conflated advantages: the sender's informational advantage and first-proposer advantage. With only modest modifications to the standard setting, this perspective makes explicit the common knowledge of the game structure and grants the receiver comparable commitment capabilities, thereby reinterpreting classic one-sided persuasion as a balanced information bargaining framework. The framework is validated through a two-stage validation-and-inference paradigm: We first demonstrate that GPT-o3 and DeepSeek-R1, out of publicly available LLMs, reliably handle standard tasks; We then apply them to persuasion scenarios to test that the outcomes align with what our information-bargaining framework suggests. All code, results, and terminal logs are publicly available at this http URL. 

---
# Object Navigation with Structure-Semantic Reasoning-Based Multi-level Map and Multimodal Decision-Making LLM 

**Authors**: Chongshang Yan, Jiaxuan He, Delun Li, Yi Yang, Wenjie Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.05896)  

**Abstract**: The zero-shot object navigation (ZSON) in unknown open-ended environments coupled with semantically novel target often suffers from the significant decline in performance due to the neglect of high-dimensional implicit scene information and the long-range target searching task. To address this, we proposed an active object navigation framework with Environmental Attributes Map (EAM) and MLLM Hierarchical Reasoning module (MHR) to improve its success rate and efficiency. EAM is constructed by reasoning observed environments with SBERT and predicting unobserved ones with Diffusion, utilizing human space regularities that underlie object-room correlations and area adjacencies. MHR is inspired by EAM to perform frontier exploration decision-making, avoiding the circuitous trajectories in long-range scenarios to improve path efficiency. Experimental results demonstrate that the EAM module achieves 64.5\% scene mapping accuracy on MP3D dataset, while the navigation task attains SPLs of 28.4\% and 26.3\% on HM3D and MP3D benchmarks respectively - representing absolute improvements of 21.4\% and 46.0\% over baseline methods. 

---
# MoA: Heterogeneous Mixture of Adapters for Parameter-Efficient Fine-Tuning of Large Language Models 

**Authors**: Jie Cao, Tianwei Lin, Hongyang He, Rolan Yan, Wenqiao Zhang, Juncheng Li, Dongping Zhang, Siliang Tang, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05928)  

**Abstract**: Recent studies integrate Low-Rank Adaptation (LoRA) and Mixture-of-Experts (MoE) to further enhance the performance of parameter-efficient fine-tuning (PEFT) methods in Large Language Model (LLM) applications. Existing methods employ \emph{homogeneous} MoE-LoRA architectures composed of LoRA experts with either similar or identical structures and capacities. However, these approaches often suffer from representation collapse and expert load imbalance, which negatively impact the potential of LLMs. To address these challenges, we propose a \emph{heterogeneous} \textbf{Mixture-of-Adapters (MoA)} approach. This method dynamically integrates PEFT adapter experts with diverse structures, leveraging their complementary representational capabilities to foster expert specialization, thereby enhancing the effective transfer of pre-trained knowledge to downstream tasks. MoA supports two variants: \textbf{(i)} \textit{Soft MoA} achieves fine-grained integration by performing a weighted fusion of all expert outputs; \textbf{(ii)} \textit{Sparse MoA} activates adapter experts sparsely based on their contribution, achieving this with negligible performance degradation. Experimental results demonstrate that heterogeneous MoA outperforms homogeneous MoE-LoRA methods in both performance and parameter efficiency. Our project is available at this https URL. 

---
# Efficient Online RFT with Plug-and-Play LLM Judges: Unlocking State-of-the-Art Performance 

**Authors**: Rudransh Agnihotri, Ananya Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2506.05748)  

**Abstract**: Reward-model training is the cost bottleneck in modern Reinforcement Learning Human Feedback (RLHF) pipelines, often requiring tens of billions of parameters and an offline preference-tuning phase. In the proposed method, a frozen, instruction-tuned 7B LLM is augmented with only a one line JSON rubric and a rank-16 LoRA adapter (affecting just 0.8% of the model's parameters), enabling it to serve as a complete substitute for the previously used heavyweight evaluation models. The plug-and-play judge achieves 96.2% accuracy on RewardBench, outperforming specialized reward networks ranging from 27B to 70B parameters. Additionally, it allows a 7B actor to outperform the top 70B DPO baseline, which scores 61.8%, by achieving 92% exact match accuracy on GSM-8K utilizing online PPO. Thorough ablations indicate that (i) six in context demonstrations deliver the majority of the zero-to-few-shot improvements (+2pp), and (ii) the LoRA effectively addresses the remaining disparity, particularly in the safety and adversarial Chat-Hard segments. The proposed model introduces HH-Rationales, a subset of 10,000 pairs from Anthropic HH-RLHF, to examine interpretability, accompanied by human generated justifications. GPT-4 scoring indicates that our LoRA judge attains approximately = 9/10 in similarity to human explanations, while zero-shot judges score around =5/10. These results indicate that the combination of prompt engineering and tiny LoRA produces a cost effective, transparent, and easily adjustable reward function, removing the offline phase while achieving new state-of-the-art outcomes for both static evaluation and online RLHF. 

---
# To Protect the LLM Agent Against the Prompt Injection Attack with Polymorphic Prompt 

**Authors**: Zhilong Wang, Neha Nagaraja, Lan Zhang, Hayretdin Bahsi, Pawan Patil, Peng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05739)  

**Abstract**: LLM agents are widely used as agents for customer support, content generation, and code assistance. However, they are vulnerable to prompt injection attacks, where adversarial inputs manipulate the model's behavior. Traditional defenses like input sanitization, guard models, and guardrails are either cumbersome or ineffective. In this paper, we propose a novel, lightweight defense mechanism called Polymorphic Prompt Assembling (PPA), which protects against prompt injection with near-zero overhead. The approach is based on the insight that prompt injection requires guessing and breaking the structure of the system prompt. By dynamically varying the structure of system prompts, PPA prevents attackers from predicting the prompt structure, thereby enhancing security without compromising performance. We conducted experiments to evaluate the effectiveness of PPA against existing attacks and compared it with other defense methods. 

---
# ScaleRTL: Scaling LLMs with Reasoning Data and Test-Time Compute for Accurate RTL Code Generation 

**Authors**: Chenhui Deng, Yun-Da Tsai, Guan-Ting Liu, Zhongzhi Yu, Haoxing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2506.05566)  

**Abstract**: Recent advances in large language models (LLMs) have enabled near-human performance on software coding benchmarks, but their effectiveness in RTL code generation remains limited due to the scarcity of high-quality training data. While prior efforts have fine-tuned LLMs for RTL tasks, they do not fundamentally overcome the data bottleneck and lack support for test-time scaling due to their non-reasoning nature. In this work, we introduce ScaleRTL, the first reasoning LLM for RTL coding that scales up both high-quality reasoning data and test-time compute. Specifically, we curate a diverse set of long chain-of-thought reasoning traces averaging 56K tokens each, resulting in a dataset of 3.5B tokens that captures rich RTL knowledge. Fine-tuning a general-purpose reasoning model on this corpus yields ScaleRTL that is capable of deep RTL reasoning. Subsequently, we further enhance the performance of ScaleRTL through a novel test-time scaling strategy that extends the reasoning process via iteratively reflecting on and self-correcting previous reasoning steps. Experimental results show that ScaleRTL achieves state-of-the-art performance on VerilogEval and RTLLM, outperforming 18 competitive baselines by up to 18.4% on VerilogEval and 12.7% on RTLLM. 

---
# Conformal Prediction Beyond the Seen: A Missing Mass Perspective for Uncertainty Quantification in Generative Models 

**Authors**: Sima Noorani, Shayan Kiyani, George Pappas, Hamed Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2506.05497)  

**Abstract**: Uncertainty quantification (UQ) is essential for safe deployment of generative AI models such as large language models (LLMs), especially in high stakes applications. Conformal prediction (CP) offers a principled uncertainty quantification framework, but classical methods focus on regression and classification, relying on geometric distances or softmax scores: tools that presuppose structured outputs. We depart from this paradigm by studying CP in a query only setting, where prediction sets must be constructed solely from finite queries to a black box generative model, introducing a new trade off between coverage, test time query budget, and informativeness. We introduce Conformal Prediction with Query Oracle (CPQ), a framework characterizing the optimal interplay between these objectives. Our finite sample algorithm is built on two core principles: one governs the optimal query policy, and the other defines the optimal mapping from queried samples to prediction sets. Remarkably, both are rooted in the classical missing mass problem in statistics. Specifically, the optimal query policy depends on the rate of decay, or the derivative, of the missing mass, for which we develop a novel estimator. Meanwhile, the optimal mapping hinges on the missing mass itself, which we estimate using Good Turing estimators. We then turn our focus to implementing our method for language models, where outputs are vast, variable, and often under specified. Fine grained experiments on three real world open ended tasks and two LLMs, show CPQ applicability to any black box LLM and highlight: (1) individual contribution of each principle to CPQ performance, and (2) CPQ ability to yield significantly more informative prediction sets than existing conformal methods for language uncertainty quantification. 

---
# PCDVQ: Enhancing Vector Quantization for Large Language Models via Polar Coordinate Decoupling 

**Authors**: Yuxuan Yue, Zukang Xu, Zhihang Yuan, Dawei Yang, Jianglong Wu, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.05432)  

**Abstract**: Large Language Models (LLMs) face significant challenges in edge deployment due to their massive parameter scale. Vector Quantization (VQ), a clustering-based quantization method, serves as a prevalent solution to this issue for its extremely low-bit (even at 2-bit) and considerable accuracy. Since a vector is a quantity in mathematics and physics that has both direction and magnitude, existing VQ works typically quantize them in a coupled manner. However, we find that direction exhibits significantly greater sensitivity to quantization compared to the magnitude. For instance, when separately clustering the directions and magnitudes of weight vectors in LLaMA-2-7B, the accuracy drop of zero-shot tasks are 46.5\% and 2.3\%, respectively. This gap even increases with the reduction of clustering centers. Further, Euclidean distance, a common metric to access vector similarities in current VQ works, places greater emphasis on reducing the magnitude error. This property is contrary to the above finding, unavoidably leading to larger quantization errors. To these ends, this paper proposes Polar Coordinate Decoupled Vector Quantization (PCDVQ), an effective and efficient VQ framework consisting of two key modules: 1) Polar Coordinate Decoupling (PCD), which transforms vectors into their polar coordinate representations and perform independent quantization of the direction and magnitude parameters.2) Distribution Aligned Codebook Construction (DACC), which optimizes the direction and magnitude codebooks in accordance with the source distribution. Experimental results show that PCDVQ outperforms baseline methods at 2-bit level by at least 1.5\% zero-shot accuracy, establishing a novel paradigm for accurate and highly compressed LLMs. 

---
# SIV-Bench: A Video Benchmark for Social Interaction Understanding and Reasoning 

**Authors**: Fanqi Kong, Weiqin Zu, Xinyu Chen, Yaodong Yang, Song-Chun Zhu, Xue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.05425)  

**Abstract**: The rich and multifaceted nature of human social interaction, encompassing multimodal cues, unobservable relations and mental states, and dynamical behavior, presents a formidable challenge for artificial intelligence. To advance research in this area, we introduce SIV-Bench, a novel video benchmark for rigorously evaluating the capabilities of Multimodal Large Language Models (MLLMs) across Social Scene Understanding (SSU), Social State Reasoning (SSR), and Social Dynamics Prediction (SDP). SIV-Bench features 2,792 video clips and 8,792 meticulously generated question-answer pairs derived from a human-LLM collaborative pipeline. It is originally collected from TikTok and YouTube, covering a wide range of video genres, presentation styles, and linguistic and cultural backgrounds. It also includes a dedicated setup for analyzing the impact of different textual cues-original on-screen text, added dialogue, or no text. Our comprehensive experiments on leading MLLMs reveal that while models adeptly handle SSU, they significantly struggle with SSR and SDP, where Relation Inference (RI) is an acute bottleneck, as further examined in our analysis. Our study also confirms the critical role of transcribed dialogue in aiding comprehension of complex social interactions. By systematically identifying current MLLMs' strengths and limitations, SIV-Bench offers crucial insights to steer the development of more socially intelligent AI. The dataset and code are available at this https URL. 

---
# Designing DSIC Mechanisms for Data Sharing in the Era of Large Language Models 

**Authors**: Seyed Moein Ayyoubzadeh, Kourosh Shahnazari, Mohammmadali Keshtparvar, MohammadAmin Fazli  

**Link**: [PDF](https://arxiv.org/pdf/2506.05379)  

**Abstract**: Training large language models (LLMs) requires vast amounts of high-quality data from institutions that face legal, privacy, and strategic constraints. Existing data procurement methods often rely on unverifiable trust or ignore heterogeneous provider costs. We introduce a mechanism-design framework for truthful, trust-minimized data sharing that ensures dominant-strategy incentive compatibility (DSIC), individual rationality, and weak budget balance, while rewarding data based on both quality and learning utility. We formalize a model where providers privately know their data cost and quality, and value arises solely from the data's contribution to model performance. Based on this, we propose the Quality-Weighted Marginal-Incentive Auction (Q-MIA), which ranks providers using a virtual cost metric and uses Myerson-style payments to ensure DSIC and budget feasibility. To support settings with limited liquidity or long-term incentives, we introduce the Marginal Utility Token (MUT), which allocates future rights based on marginal contributions. We unify these in Mixed-MIA, a hybrid mechanism balancing upfront payments and deferred rewards. All mechanisms support verifiable, privacy-preserving implementation. Theoretically and empirically, they outperform volume-based and trust-based baselines, eliciting higher-quality data under budget constraints while remaining robust to misreporting and collusion. This establishes a principled foundation for sustainable and fair data markets for future LLMs. 

---
# Speaking images. A novel framework for the automated self-description of artworks 

**Authors**: Valentine Bernasconi, Gustavo Marfia  

**Link**: [PDF](https://arxiv.org/pdf/2506.05368)  

**Abstract**: Recent breakthroughs in generative AI have opened the door to new research perspectives in the domain of art and cultural heritage, where a large number of artifacts have been digitized. There is a need for innovation to ease the access and highlight the content of digital collections. Such innovations develop into creative explorations of the digital image in relation to its malleability and contemporary interpretation, in confrontation to the original historical object. Based on the concept of the autonomous image, we propose a new framework towards the production of self-explaining cultural artifacts using open-source large-language, face detection, text-to-speech and audio-to-animation models. The goal is to start from a digitized artwork and to automatically assemble a short video of the latter where the main character animates to explain its content. The whole process questions cultural biases encapsulated in large-language models, the potential of digital images and deepfakes of artworks for educational purposes, along with concerns of the field of art history regarding such creative diversions. 

---
# Gen4D: Synthesizing Humans and Scenes in the Wild 

**Authors**: Jerrin Bright, Zhibo Wang, Yuhao Chen, Sirisha Rambhatla, John Zelek, David Clausi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05397)  

**Abstract**: Lack of input data for in-the-wild activities often results in low performance across various computer vision tasks. This challenge is particularly pronounced in uncommon human-centric domains like sports, where real-world data collection is complex and impractical. While synthetic datasets offer a promising alternative, existing approaches typically suffer from limited diversity in human appearance, motion, and scene composition due to their reliance on rigid asset libraries and hand-crafted rendering pipelines. To address this, we introduce Gen4D, a fully automated pipeline for generating diverse and photorealistic 4D human animations. Gen4D integrates expert-driven motion encoding, prompt-guided avatar generation using diffusion-based Gaussian splatting, and human-aware background synthesis to produce highly varied and lifelike human sequences. Based on Gen4D, we present SportPAL, a large-scale synthetic dataset spanning three sports: baseball, icehockey, and soccer. Together, Gen4D and SportPAL provide a scalable foundation for constructing synthetic datasets tailored to in-the-wild human-centric vision tasks, with no need for manual 3D modeling or scene design. 

---
# SmoothRot: Combining Channel-Wise Scaling and Rotation for Quantization-Friendly LLMs 

**Authors**: Patrik Czakó, Gábor Kertész, Sándor Szénási  

**Link**: [PDF](https://arxiv.org/pdf/2506.05413)  

**Abstract**: We present SmoothRot, a novel post-training quantization technique to enhance the efficiency of 4-bit quantization in Large Language Models (LLMs). SmoothRot addresses the critical challenge of massive activation outliers, by integrating channel-wise scaling with Hadamard transformations. Our technique effectively transforms extreme outliers into quantization-friendly activations, significantly improving quantization accuracy. Experiments conducted on popular LLMs (LLaMA2 7B, LLaMA3.1 8B, and Mistral 7B) demonstrate that SmoothRot consistently reduces the performance gap between quantized and FP16 models by approximately 10-30\% across language generation and zero-shot reasoning tasks, without introducing additional inference latency. Code is available at this https URL. 

---
# Diffusion with a Linguistic Compass: Steering the Generation of Clinically Plausible Future sMRI Representations for Early MCI Conversion Prediction 

**Authors**: Zhihao Tang, Chaozhuo Li, Litian Zhang, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05428)  

**Abstract**: Early prediction of Mild Cognitive Impairment (MCI) conversion is hampered by a trade-off between immediacy--making fast predictions from a single baseline sMRI--and accuracy--leveraging longitudinal scans to capture disease progression. We propose MCI-Diff, a diffusion-based framework that synthesizes clinically plausible future sMRI representations directly from baseline data, achieving both real-time risk assessment and high predictive performance. First, a multi-task sequence reconstruction strategy trains a shared denoising network on interpolation and extrapolation tasks to handle irregular follow-up sampling and learn robust latent trajectories. Second, an LLM-driven "linguistic compass" is introduced for clinical plausibility sampling: generated feature candidates are quantized, tokenized, and scored by a fine-tuned language model conditioned on expected structural biomarkers, guiding autoregressive generation toward realistic disease patterns. Experiments on ADNI and AIBL cohorts show that MCI-Diff outperforms state-of-the-art baselines, improving early conversion accuracy by 5-12%. 

---
# Can ChatGPT Perform Image Splicing Detection? A Preliminary Study 

**Authors**: Souradip Nath  

**Link**: [PDF](https://arxiv.org/pdf/2506.05358)  

**Abstract**: Multimodal Large Language Models (MLLMs) like GPT-4V are capable of reasoning across text and image modalities, showing promise in a variety of complex vision-language tasks. In this preliminary study, we investigate the out-of-the-box capabilities of GPT-4V in the domain of image forensics, specifically, in detecting image splicing manipulations. Without any task-specific fine-tuning, we evaluate GPT-4V using three prompting strategies: Zero-Shot (ZS), Few-Shot (FS), and Chain-of-Thought (CoT), applied over a curated subset of the CASIA v2.0 splicing dataset.
Our results show that GPT-4V achieves competitive detection performance in zero-shot settings (more than 85% accuracy), with CoT prompting yielding the most balanced trade-off across authentic and spliced images. Qualitative analysis further reveals that the model not only detects low-level visual artifacts but also draws upon real-world contextual knowledge such as object scale, semantic consistency, and architectural facts, to identify implausible composites. While GPT-4V lags behind specialized state-of-the-art splicing detection models, its generalizability, interpretability, and encyclopedic reasoning highlight its potential as a flexible tool in image forensics. 

---
