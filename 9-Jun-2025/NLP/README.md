# AdvSumm: Adversarial Training for Bias Mitigation in Text Summarization 

**Authors**: Mukur Gupta, Nikhil Reddy Varimalla, Nicholas Deas, Melanie Subbiah, Kathleen McKeown  

**Link**: [PDF](https://arxiv.org/pdf/2506.06273)  

**Abstract**: Large Language Models (LLMs) have achieved impressive performance in text summarization and are increasingly deployed in real-world applications. However, these systems often inherit associative and framing biases from pre-training data, leading to inappropriate or unfair outputs in downstream tasks. In this work, we present AdvSumm (Adversarial Summarization), a domain-agnostic training framework designed to mitigate bias in text summarization through improved generalization. Inspired by adversarial robustness, AdvSumm introduces a novel Perturber component that applies gradient-guided perturbations at the embedding level of Sequence-to-Sequence models, enhancing the model's robustness to input variations. We empirically demonstrate that AdvSumm effectively reduces different types of bias in summarization-specifically, name-nationality bias and political framing bias-without compromising summarization quality. Compared to standard transformers and data augmentation techniques like back-translation, AdvSumm achieves stronger bias mitigation performance across benchmark datasets. 

---
# Cartridges: Lightweight and general-purpose long context representations via self-study 

**Authors**: Sabri Eyuboglu, Ryan Ehrlich, Simran Arora, Neel Guha, Dylan Zinsley, Emily Liu, Will Tennien, Atri Rudra, James Zou, Azalia Mirhoseini, Christopher Re  

**Link**: [PDF](https://arxiv.org/pdf/2506.06266)  

**Abstract**: Large language models are often used to answer queries grounded in large text corpora (e.g. codebases, legal documents, or chat histories) by placing the entire corpus in the context window and leveraging in-context learning (ICL). Although current models support contexts of 100K-1M tokens, this setup is costly to serve because the memory consumption of the KV cache scales with input length. We explore an alternative: training a smaller KV cache offline on each corpus. At inference time, we load this trained KV cache, which we call a Cartridge, and decode a response. Critically, the cost of training a Cartridge can be amortized across all the queries referencing the same corpus. However, we find that the naive approach of training the Cartridge with next-token prediction on the corpus is not competitive with ICL. Instead, we propose self-study, a training recipe in which we generate synthetic conversations about the corpus and train the Cartridge with a context-distillation objective. We find that Cartridges trained with self-study replicate the functionality of ICL, while being significantly cheaper to serve. On challenging long-context benchmarks, Cartridges trained with self-study match ICL performance while using 38.6x less memory and enabling 26.4x higher throughput. Self-study also extends the model's effective context length (e.g. from 128k to 484k tokens on MTOB) and surprisingly, leads to Cartridges that can be composed at inference time without retraining. 

---
# Bridging External and Parametric Knowledge: Mitigating Hallucination of LLMs with Shared-Private Semantic Synergy in Dual-Stream Knowledge 

**Authors**: Yi Sui, Chaozhuo Li, Chen Zhang, Dawei song, Qiuchi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06240)  

**Abstract**: Retrieval-augmented generation (RAG) is a cost-effective approach to mitigate the hallucination of Large Language Models (LLMs) by incorporating the retrieved external knowledge into the generation process. However, external knowledge may conflict with the parametric knowledge of LLMs. Furthermore, current LLMs lack inherent mechanisms for resolving such knowledge conflicts, making traditional RAG methods suffer from degraded performance and stability. Thus, we propose a Dual-Stream Knowledge-Augmented Framework for Shared-Private Semantic Synergy (DSSP-RAG). Central to the framework is a novel approach that refines self-attention into a mixed-attention, distinguishing shared and private semantics for a controlled internal-external knowledge integration. To effectively facilitate DSSP in RAG, we further introduce an unsupervised hallucination detection method based on cognitive uncertainty, ensuring the necessity of introducing knowledge, and an Energy Quotient (EQ) based on attention difference matrices to reduce noise in the retrieved external knowledge. Extensive experiments on benchmark datasets show that DSSP-RAG can effectively resolve conflicts and enhance the complementarity of dual-stream knowledge, leading to superior performance over strong baselines. 

---
# Explaining Matters: Leveraging Definitions and Semantic Expansion for Sexism Detection 

**Authors**: Sahrish Khan, Arshad Jhumka, Gabriele Pergola  

**Link**: [PDF](https://arxiv.org/pdf/2506.06238)  

**Abstract**: The detection of sexism in online content remains an open problem, as harmful language disproportionately affects women and marginalized groups. While automated systems for sexism detection have been developed, they still face two key challenges: data sparsity and the nuanced nature of sexist language. Even in large, well-curated datasets like the Explainable Detection of Online Sexism (EDOS), severe class imbalance hinders model generalization. Additionally, the overlapping and ambiguous boundaries of fine-grained categories introduce substantial annotator disagreement, reflecting the difficulty of interpreting nuanced expressions of sexism. To address these challenges, we propose two prompt-based data augmentation techniques: Definition-based Data Augmentation (DDA), which leverages category-specific definitions to generate semantically-aligned synthetic examples, and Contextual Semantic Expansion (CSE), which targets systematic model errors by enriching examples with task-specific semantic features. To further improve reliability in fine-grained classification, we introduce an ensemble strategy that resolves prediction ties by aggregating complementary perspectives from multiple language models. Our experimental evaluation on the EDOS dataset demonstrates state-of-the-art performance across all tasks, with notable improvements of macro F1 by 1.5 points for binary classification (Task A) and 4.1 points for fine-grained classification (Task C). 

---
# Can Theoretical Physics Research Benefit from Language Agents? 

**Authors**: Sirui Lu, Zhijing Jin, Terry Jingchen Zhang, Pavel Kos, J. Ignacio Cirac, Bernhard Schölkopf  

**Link**: [PDF](https://arxiv.org/pdf/2506.06214)  

**Abstract**: Large Language Models (LLMs) are rapidly advancing across diverse domains, yet their application in theoretical physics research is not yet mature. This position paper argues that LLM agents can potentially help accelerate theoretical, computational, and applied physics when properly integrated with domain knowledge and toolbox. We analyze current LLM capabilities for physics -- from mathematical reasoning to code generation -- identifying critical gaps in physical intuition, constraint satisfaction, and reliable reasoning. We envision future physics-specialized LLMs that could handle multimodal data, propose testable hypotheses, and design experiments. Realizing this vision requires addressing fundamental challenges: ensuring physical consistency, and developing robust verification methods. We call for collaborative efforts between physics and AI communities to help advance scientific discovery in physics. 

---
# PuzzleWorld: A Benchmark for Multimodal, Open-Ended Reasoning in Puzzlehunts 

**Authors**: Hengzhi Li, Brendon Jiang, Alexander Naehu, Regan Song, Justin Zhang, Megan Tjandrasuwita, Chanakya Ekbote, Steven-Shine Chen, Adithya Balachandran, Wei Dai, Rebecca Chang, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06211)  

**Abstract**: Puzzlehunts are a genre of complex, multi-step puzzles lacking well-defined problem definitions. In contrast to conventional reasoning benchmarks consisting of tasks with clear instructions, puzzlehunts require models to discover the underlying problem structure from multimodal evidence and iterative reasoning, mirroring real-world domains such as scientific discovery, exploratory data analysis, or investigative problem-solving. Despite recent progress in foundation models, their performance on such open-ended settings remains largely untested. In this paper, we introduce PuzzleWorld, a large-scale benchmark of 667 puzzlehunt-style problems designed to assess step-by-step, open-ended, and creative multimodal reasoning. Each puzzle is annotated with the final solution, detailed reasoning traces, and cognitive skill labels, enabling holistic benchmarking and fine-grained diagnostic analysis. Most state-of-the-art models achieve only 1-2% final answer accuracy, with the best model solving only 14% of puzzles and reaching 40% stepwise accuracy. To demonstrate the value of our reasoning annotations, we show that fine-tuning a small model on reasoning traces improves stepwise reasoning from 4% to 11%, while training on final answers alone degrades performance to near zero. Our error analysis reveals that current models exhibit myopic reasoning, are bottlenecked by the limitations of language-based inference, and lack sketching capabilities crucial for visual and spatial reasoning. We release PuzzleWorld at this https URL to support future work on building more general, open-ended, and creative reasoning systems. 

---
# Building Models of Neurological Language 

**Authors**: Henry Watkins  

**Link**: [PDF](https://arxiv.org/pdf/2506.06208)  

**Abstract**: This report documents the development and evaluation of domain-specific language models for neurology. Initially focused on building a bespoke model, the project adapted to rapid advances in open-source and commercial medical LLMs, shifting toward leveraging retrieval-augmented generation (RAG) and representational models for secure, local deployment. Key contributions include the creation of neurology-specific datasets (case reports, QA sets, textbook-derived data), tools for multi-word expression extraction, and graph-based analyses of medical terminology. The project also produced scripts and Docker containers for local hosting. Performance metrics and graph community results are reported, with future possible work open for multimodal models using open-source architectures like phi-4. 

---
# Detecting Voice Phishing with Precision: Fine-Tuning Small Language Models 

**Authors**: Ju Yong Sim, Seong Hwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.06180)  

**Abstract**: We develop a voice phishing (VP) detector by fine-tuning Llama3, a representative open-source, small language model (LM). In the prompt, we provide carefully-designed VP evaluation criteria and apply the Chain-of-Thought (CoT) technique. To evaluate the robustness of LMs and highlight differences in their performance, we construct an adversarial test dataset that places the models under challenging conditions. Moreover, to address the lack of VP transcripts, we create transcripts by referencing existing or new types of VP techniques. We compare cases where evaluation criteria are included, the CoT technique is applied, or both are used together. In the experiment, our results show that the Llama3-8B model, fine-tuned with a dataset that includes a prompt with VP evaluation criteria, yields the best performance among small LMs and is comparable to that of a GPT-4-based VP detector. These findings indicate that incorporating human expert knowledge into the prompt is more effective than using the CoT technique for small LMs in VP detection. 

---
# Does It Run and Is That Enough? Revisiting Text-to-Chart Generation with a Multi-Agent Approach 

**Authors**: James Ford, Anthony Rios  

**Link**: [PDF](https://arxiv.org/pdf/2506.06175)  

**Abstract**: Large language models can translate natural-language chart descriptions into runnable code, yet approximately 15\% of the generated scripts still fail to execute, even after supervised fine-tuning and reinforcement learning. We investigate whether this persistent error rate stems from model limitations or from reliance on a single-prompt design. To explore this, we propose a lightweight multi-agent pipeline that separates drafting, execution, repair, and judgment, using only an off-the-shelf GPT-4o-mini model. On the \textsc{Text2Chart31} benchmark, our system reduces execution errors to 4.5\% within three repair iterations, outperforming the strongest fine-tuned baseline by nearly 5 percentage points while requiring significantly less compute. Similar performance is observed on the \textsc{ChartX} benchmark, with an error rate of 4.6\%, demonstrating strong generalization. Under current benchmarks, execution success appears largely solved. However, manual review reveals that 6 out of 100 sampled charts contain hallucinations, and an LLM-based accessibility audit shows that only 33.3\% (\textsc{Text2Chart31}) and 7.2\% (\textsc{ChartX}) of generated charts satisfy basic colorblindness guidelines. These findings suggest that future work should shift focus from execution reliability toward improving chart aesthetics, semantic fidelity, and accessibility. 

---
# semantic-features: A User-Friendly Tool for Studying Contextual Word Embeddings in Interpretable Semantic Spaces 

**Authors**: Jwalanthi Ranganathan, Rohan Jha, Kanishka Misra, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2506.06169)  

**Abstract**: We introduce semantic-features, an extensible, easy-to-use library based on Chronis et al. (2023) for studying contextualized word embeddings of LMs by projecting them into interpretable spaces. We apply this tool in an experiment where we measure the contextual effect of the choice of dative construction (prepositional or double object) on the semantic interpretation of utterances (Bresnan, 2007). Specifically, we test whether "London" in "I sent London the letter." is more likely to be interpreted as an animate referent (e.g., as the name of a person) than in "I sent the letter to London." To this end, we devise a dataset of 450 sentence pairs, one in each dative construction, with recipients being ambiguous with respect to person-hood vs. place-hood. By applying semantic-features, we show that the contextualized word embeddings of three masked language models show the expected sensitivities. This leaves us optimistic about the usefulness of our tool. 

---
# Let's CONFER: A Dataset for Evaluating Natural Language Inference Models on CONditional InFERence and Presupposition 

**Authors**: Tara Azin, Daniel Dumitrescu, Diana Inkpen, Raj Singh  

**Link**: [PDF](https://arxiv.org/pdf/2506.06133)  

**Abstract**: Natural Language Inference (NLI) is the task of determining whether a sentence pair represents entailment, contradiction, or a neutral relationship. While NLI models perform well on many inference tasks, their ability to handle fine-grained pragmatic inferences, particularly presupposition in conditionals, remains underexplored. In this study, we introduce CONFER, a novel dataset designed to evaluate how NLI models process inference in conditional sentences. We assess the performance of four NLI models, including two pre-trained models, to examine their generalization to conditional reasoning. Additionally, we evaluate Large Language Models (LLMs), including GPT-4o, LLaMA, Gemma, and DeepSeek-R1, in zero-shot and few-shot prompting settings to analyze their ability to infer presuppositions with and without prior context. Our findings indicate that NLI models struggle with presuppositional reasoning in conditionals, and fine-tuning on existing NLI datasets does not necessarily improve their performance. 

---
# Phonetically-Augmented Discriminative Rescoring for Voice Search Error Correction 

**Authors**: Christophe Van Gysel, Maggie Wu, Lyan Verwimp, Caglar Tirkaz, Marco Bertola, Zhihong Lei, Youssef Oualil  

**Link**: [PDF](https://arxiv.org/pdf/2506.06117)  

**Abstract**: End-to-end (E2E) Automatic Speech Recognition (ASR) models are trained using paired audio-text samples that are expensive to obtain, since high-quality ground-truth data requires human annotators. Voice search applications, such as digital media players, leverage ASR to allow users to search by voice as opposed to an on-screen keyboard. However, recent or infrequent movie titles may not be sufficiently represented in the E2E ASR system's training data, and hence, may suffer poor recognition.
In this paper, we propose a phonetic correction system that consists of (a) a phonetic search based on the ASR model's output that generates phonetic alternatives that may not be considered by the E2E system, and (b) a rescorer component that combines the ASR model recognition and the phonetic alternatives, and select a final system output.
We find that our approach improves word error rate between 4.4 and 7.6% relative on benchmarks of popular movie titles over a series of competitive baselines. 

---
# Bridging the Gap: In-Context Learning for Modeling Human Disagreement 

**Authors**: Benedetta Muscato, Yue Li, Gizem Gezici, Zhixue Zhao, Fosca Giannotti  

**Link**: [PDF](https://arxiv.org/pdf/2506.06113)  

**Abstract**: Large Language Models (LLMs) have shown strong performance on NLP classification tasks. However, they typically rely on aggregated labels-often via majority voting-which can obscure the human disagreement inherent in subjective annotations. This study examines whether LLMs can capture multiple perspectives and reflect annotator disagreement in subjective tasks such as hate speech and offensive language detection. We use in-context learning (ICL) in zero-shot and few-shot settings, evaluating four open-source LLMs across three label modeling strategies: aggregated hard labels, and disaggregated hard and soft labels. In few-shot prompting, we assess demonstration selection methods based on textual similarity (BM25, PLM-based), annotation disagreement (entropy), a combined ranking, and example ordering strategies (random vs. curriculum-based). Results show that multi-perspective generation is viable in zero-shot settings, while few-shot setups often fail to capture the full spectrum of human judgments. Prompt design and demonstration selection notably affect performance, though example ordering has limited impact. These findings highlight the challenges of modeling subjectivity with LLMs and the importance of building more perspective-aware, socially intelligent models. 

---
# Reinforcing Code Generation: Improving Text-to-SQL with Execution-Based Learning 

**Authors**: Atharv Kulkarni, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.06093)  

**Abstract**: In this work, we study the problem of code generation with a large language model (LLM), with a focus on generating SQL queries from natural language questions. We ask: Instead of using supervised fine tuning with text-code pairs, can we tune a model by having it interact with a database engine? We frame this problem as a reinforcement learning problem where the model receives execution-based feedback from the environment in the form of scalar rewards. These rewards penalize execution failures and assign positive values when a query returns a correct answer. We use the rewards within the Group Relative Policy Optimization (GRPO) framework. We use a tabular reasoning benchmark to test and evaluate our findings. We find that with only weak supervision in the form of question-answer pairs, RL-tuning improves the accuracy of model generated SQL code from 31.49 to 49.83 while reducing error percentage from 25.43% to 14.71%. This improvement allowed the model nearly match the performance performance to the larger SQLCoder-70B model. Our work demonstrates the potential of using execution-based feedback to improve symbolic reasoning capabilities of LLMs. 

---
# MIRIAD: Augmenting LLMs with millions of medical query-response pairs 

**Authors**: Qinyue Zheng, Salman Abdullah, Sam Rawal, Cyril Zakka, Sophie Ostmeier, Maximilian Purk, Eduardo Reis, Eric J. Topol, Jure Leskovec, Michael Moor  

**Link**: [PDF](https://arxiv.org/pdf/2506.06091)  

**Abstract**: LLMs are bound to transform healthcare with advanced decision support and flexible chat assistants. However, LLMs are prone to generate inaccurate medical content. To ground LLMs in high-quality medical knowledge, LLMs have been equipped with external knowledge via RAG, where unstructured medical knowledge is split into small text chunks that can be selectively retrieved and integrated into the LLMs context. Yet, existing RAG pipelines rely on raw, unstructured medical text, which can be noisy, uncurated and difficult for LLMs to effectively leverage. Systematic approaches to organize medical knowledge to best surface it to LLMs are generally lacking. To address these challenges, we introduce MIRIAD, a large-scale, curated corpus of 5,821,948 medical QA pairs, each rephrased from and grounded in a passage from peer-reviewed medical literature using a semi-automated pipeline combining LLM generation, filtering, grounding, and human annotation. Unlike prior medical corpora, which rely on unstructured text, MIRIAD encapsulates web-scale medical knowledge in an operationalized query-response format, which enables more targeted retrieval. Experiments on challenging medical QA benchmarks show that augmenting LLMs with MIRIAD improves accuracy up to 6.7% compared to unstructured RAG baselines with the same source corpus and with the same amount of retrieved text. Moreover, MIRIAD improved the ability of LLMs to detect medical hallucinations by 22.5 to 37% (increase in F1 score). We further introduce MIRIAD-Atlas, an interactive map of MIRIAD spanning 56 medical disciplines, enabling clinical users to visually explore, search, and refine medical knowledge. MIRIAD promises to unlock a wealth of down-stream applications, including medical information retrievers, enhanced RAG applications, and knowledge-grounded chat interfaces, which ultimately enables more reliable LLM applications in healthcare. 

---
# Zero-Shot Detection of LLM-Generated Code via Approximated Task Conditioning 

**Authors**: Maor Ashkenazi, Ofir Brenner, Tal Furman Shohet, Eran Treister  

**Link**: [PDF](https://arxiv.org/pdf/2506.06069)  

**Abstract**: Detecting Large Language Model (LLM)-generated code is a growing challenge with implications for security, intellectual property, and academic integrity. We investigate the role of conditional probability distributions in improving zero-shot LLM-generated code detection, when considering both the code and the corresponding task prompt that generated it. Our key insight is that when evaluating the probability distribution of code tokens using an LLM, there is little difference between LLM-generated and human-written code. However, conditioning on the task reveals notable differences. This contrasts with natural language text, where differences exist even in the unconditional distributions. Leveraging this, we propose a novel zero-shot detection approach that approximates the original task used to generate a given code snippet and then evaluates token-level entropy under the approximated task conditioning (ATC). We further provide a mathematical intuition, contextualizing our method relative to previous approaches. ATC requires neither access to the generator LLM nor the original task prompts, making it practical for real-world applications. To the best of our knowledge, it achieves state-of-the-art results across benchmarks and generalizes across programming languages, including Python, CPP, and Java. Our findings highlight the importance of task-level conditioning for LLM-generated code detection. The supplementary materials and code are available at this https URL, including the dataset gathering implementation, to foster further research in this area. 

---
# Simple Yet Effective: Extracting Private Data Across Clients in Federated Fine-Tuning of Large Language Models 

**Authors**: Yingqi Hu, Zhuo Zhang, Jingyuan Zhang, Lizhen Qu, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06060)  

**Abstract**: Federated fine-tuning of large language models (FedLLMs) presents a promising approach for achieving strong model performance while preserving data privacy in sensitive domains. However, the inherent memorization ability of LLMs makes them vulnerable to training data extraction attacks. To investigate this risk, we introduce simple yet effective extraction attack algorithms specifically designed for FedLLMs. In contrast to prior "verbatim" extraction attacks, which assume access to fragments from all training data, our approach operates under a more realistic threat model, where the attacker only has access to a single client's data and aims to extract previously unseen personally identifiable information (PII) from other clients. This requires leveraging contextual prefixes held by the attacker to generalize across clients. To evaluate the effectiveness of our approaches, we propose two rigorous metrics-coverage rate and efficiency-and extend a real-world legal dataset with PII annotations aligned with CPIS, GDPR, and CCPA standards, achieving 89.9% human-verified precision. Experimental results show that our method can extract up to 56.57% of victim-exclusive PII, with "Address," "Birthday," and "Name" being the most vulnerable categories. Our findings underscore the pressing need for robust defense strategies and contribute a new benchmark and evaluation framework for future research in privacy-preserving federated learning. 

---
# Hey, That's My Data! Label-Only Dataset Inference in Large Language Models 

**Authors**: Chen Xiong, Zihao Wang, Rui Zhu, Tsung-Yi Ho, Pin-Yu Chen, Jingwei Xiong, Haixu Tang, Lucila Ohno-Machado  

**Link**: [PDF](https://arxiv.org/pdf/2506.06057)  

**Abstract**: Large Language Models (LLMs) have revolutionized Natural Language Processing by excelling at interpreting, reasoning about, and generating human language. However, their reliance on large-scale, often proprietary datasets poses a critical challenge: unauthorized usage of such data can lead to copyright infringement and significant financial harm. Existing dataset-inference methods typically depend on log probabilities to detect suspicious training material, yet many leading LLMs have begun withholding or obfuscating these signals. This reality underscores the pressing need for label-only approaches capable of identifying dataset membership without relying on internal model logits.
We address this gap by introducing CatShift, a label-only dataset-inference framework that capitalizes on catastrophic forgetting: the tendency of an LLM to overwrite previously learned knowledge when exposed to new data. If a suspicious dataset was previously seen by the model, fine-tuning on a portion of it triggers a pronounced post-tuning shift in the model's outputs; conversely, truly novel data elicits more modest changes. By comparing the model's output shifts for a suspicious dataset against those for a known non-member validation set, we statistically determine whether the suspicious set is likely to have been part of the model's original training corpus. Extensive experiments on both open-source and API-based LLMs validate CatShift's effectiveness in logit-inaccessible settings, offering a robust and practical solution for safeguarding proprietary data. 

---
# MATP-BENCH: Can MLLM Be a Good Automated Theorem Prover for Multimodal Problems? 

**Authors**: Zhitao He, Zongwei Lyu, Dazhong Chen, Dadi Guo, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2506.06034)  

**Abstract**: Numerous theorems, such as those in geometry, are often presented in multimodal forms (e.g., diagrams). Humans benefit from visual reasoning in such settings, using diagrams to gain intuition and guide the proof process. Modern Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in solving a wide range of mathematical problems. However, the potential of MLLMs as Automated Theorem Provers (ATPs), specifically in the multimodal domain, remains underexplored. In this paper, we introduce the Multimodal Automated Theorem Proving benchmark (MATP-BENCH), a new Multimodal, Multi-level, and Multi-language benchmark designed to evaluate MLLMs in this role as multimodal automated theorem provers. MATP-BENCH consists of 1056 multimodal theorems drawn from high school, university, and competition-level mathematics. All these multimodal problems are accompanied by formalizations in Lean 4, Coq and Isabelle, thus making the benchmark compatible with a wide range of theorem-proving frameworks. MATP-BENCH requires models to integrate sophisticated visual understanding with mastery of a broad spectrum of mathematical knowledge and rigorous symbolic reasoning to generate formal proofs. We use MATP-BENCH to evaluate a variety of advanced multimodal language models. Existing methods can only solve a limited number of the MATP-BENCH problems, indicating that this benchmark poses an open challenge for research on automated theorem proving. 

---
# Large Language Models are Demonstration Pre-Selectors for Themselves 

**Authors**: Jiarui Jin, Yuwei Wu, Haoxuan Li, Xiaoting He, Weinan Zhang, Yiming Yang, Yong Yu, Jun Wang, Mengyue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06033)  

**Abstract**: In-context learning (ICL) with large language models (LLMs) delivers strong few-shot performance by choosing few-shot demonstrations from the entire training data. However, existing ICL methods, which rely on similarity or diversity scores to choose demonstrations, incur high computational costs due to repeatedly retrieval from large-scale datasets for each query. To this end, we propose FEEDER (FEw yet Essential Demonstration prE-selectoR), a novel pre-selection framework that identifies a representative subset of demonstrations containing the most representative examples in the training data, tailored to specific LLMs. To construct this subset, we introduce the "sufficiency" and "necessity" metrics in the pre-selection stage and design a tree-based algorithm to identify representative examples efficiently. Once pre-selected, this representative subset can effectively replace the full training data, improving efficiency while maintaining comparable performance in ICL. Additionally, our pre-selected subset also benefits fine-tuning LLMs, where we introduce a bi-level optimization method that enhances training efficiency without sacrificing performance. Experiments with LLMs ranging from 300M to 8B parameters show that FEEDER can reduce training data size by over 20% while maintaining performance and seamlessly integrating with various downstream demonstration selection strategies in ICL. 

---
# When to Trust Context: Self-Reflective Debates for Context Reliability 

**Authors**: Zeqi Zhou, Fang Wu, Shayan Talaei, Haokai Zhao, Cheng Meixin, Tinson Xu, Amin Saberi, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06020)  

**Abstract**: Large language models frequently encounter conflicts between their parametric knowledge and contextual input, often resulting in factual inconsistencies or hallucinations. We propose Self-Reflective Debate for Contextual Reliability (SR-DCR), a lightweight framework that integrates token-level self-confidence with an asymmetric multi-agent debate to adjudicate such conflicts. A critic, deprived of context, challenges a defender who argues from the given passage; a judge model evaluates the debate and determines the context's reliability. The final answer is selected by combining the verdict with model confidence. Experiments on the ClashEval benchmark demonstrate that SR-DCR consistently enhances robustness to misleading context while maintaining accuracy on trustworthy inputs, outperforming both classical debate and confidence-only baselines with minimal computational overhead. The code is available at this https URL. 

---
# AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search 

**Authors**: Yu Li, Lehui Li, Zhihao Wu, Qingmin Liao, Jianye Hao, Kun Shao, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06017)  

**Abstract**: Large language model (LLM) agents have demonstrated strong capabilities across diverse domains. However, designing high-performing agentic systems remains challenging. Existing agent search methods suffer from three major limitations: (1) an emphasis on optimizing agentic workflows while under-utilizing proven human-designed components such as memory, planning, and tool use; (2) high evaluation costs, as each newly generated agent must be fully evaluated on benchmarks; and (3) inefficient search in large search space. In this work, we introduce a comprehensive framework to address these challenges. First, We propose a hierarchical search space that jointly models agentic workflow and composable functional components, enabling richer agentic system designs. Building on this structured design space, we introduce a predictive value model that estimates agent performance given agentic system and task description, allowing for efficient, low-cost evaluation during the search process. Finally, we present a hierarchical Monte Carlo Tree Search (MCTS) strategy informed by uncertainty to guide the search. Experiments on seven benchmarks, covering embodied, math, web, tool, and game, show that our method achieves an average performance gain of 8.34\% over state-of-the-art baselines and exhibits faster search progress with steeper improvement trajectories. Code repo is available at this https URL. 

---
# Unlocking Recursive Thinking of LLMs: Alignment via Refinement 

**Authors**: Haoke Zhang, Xiaobo Liang, Cunxiang Wang, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06009)  

**Abstract**: The OpenAI o1-series models have demonstrated that leveraging long-form Chain of Thought (CoT) can substantially enhance performance. However, the recursive thinking capabilities of Large Language Models (LLMs) remain limited, particularly in the absence of expert-curated data for distillation. In this paper, we propose \textbf{AvR}: \textbf{Alignment via Refinement}, a novel method aimed at unlocking the potential of LLMs for recursive reasoning through long-form CoT. AvR introduces a refinement process that integrates criticism and improvement actions, guided by differentiable learning techniques to optimize \textbf{refinement-aware rewards}. As a result, the synthesized multi-round data can be organized as a long refinement thought, further enabling test-time scaling. Experimental results show that AvR significantly outperforms conventional preference optimization methods. Notably, with only 3k synthetic samples, our method boosts the performance of the LLaMA-3-8B-Instruct model by over 20\% in win rate on AlpacaEval 2.0. Our code is available at Github (this https URL). 

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
# Tau-Eval: A Unified Evaluation Framework for Useful and Private Text Anonymization 

**Authors**: Gabriel Loiseau, Damien Sileo, Damien Riquet, Maxime Meyer, Marc Tommasi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05979)  

**Abstract**: Text anonymization is the process of removing or obfuscating information from textual data to protect the privacy of individuals. This process inherently involves a complex trade-off between privacy protection and information preservation, where stringent anonymization methods can significantly impact the text's utility for downstream applications. Evaluating the effectiveness of text anonymization proves challenging from both privacy and utility perspectives, as there is no universal benchmark that can comprehensively assess anonymization techniques across diverse, and sometimes contradictory contexts. We present Tau-Eval, an open-source framework for benchmarking text anonymization methods through the lens of privacy and utility task sensitivity. A Python library, code, documentation and tutorials are publicly available. 

---
# LTG at SemEval-2025 Task 10: Optimizing Context for Classification of Narrative Roles 

**Authors**: Egil Rønningstad, Gaurav Negi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05976)  

**Abstract**: Our contribution to the SemEval 2025 shared task 10, subtask 1 on entity framing, tackles the challenge of providing the necessary segments from longer documents as context for classification with a masked language model. We show that a simple entity-oriented heuristics for context selection can enable text classification using models with limited context window. Our context selection approach and the XLM-RoBERTa language model is on par with, or outperforms, Supervised Fine-Tuning with larger generative language models. 

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
# NameTag 3: A Tool and a Service for Multilingual/Multitagset NER 

**Authors**: Jana Straková, Milan Straka  

**Link**: [PDF](https://arxiv.org/pdf/2506.05949)  

**Abstract**: We introduce NameTag 3, an open-source tool and cloud-based web service for multilingual, multidataset, and multitagset named entity recognition (NER), supporting both flat and nested entities. NameTag 3 achieves state-of-the-art results on 21 test datasets in 15 languages and remains competitive on the rest, even against larger models. It is available as a command-line tool and as a cloud-based service, enabling use without local installation. NameTag 3 web service currently provides flat NER for 17 languages, trained on 21 corpora and three NE tagsets, all powered by a single 355M-parameter fine-tuned model; and nested NER for Czech, powered by a 126M fine-tuned model. The source code is licensed under open-source MPL 2.0, while the models are distributed under non-commercial CC BY-NC-SA 4.0. Documentation is available at this https URL, source code at this https URL, and trained models via this https URL. The REST service and the web application can be found at this https URL. A demonstration video is available at this https URL. 

---
# IntentionESC: An Intention-Centered Framework for Enhancing Emotional Support in Dialogue Systems 

**Authors**: Xinjie Zhang, Wenxuan Wang, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.05947)  

**Abstract**: In emotional support conversations, unclear intentions can lead supporters to employ inappropriate strategies, inadvertently imposing their expectations or solutions on the seeker. Clearly defined intentions are essential for guiding both the supporter's motivations and the overall emotional support process. In this paper, we propose the Intention-centered Emotional Support Conversation (IntentionESC) framework, which defines the possible intentions of supporters in emotional support conversations, identifies key emotional state aspects for inferring these intentions, and maps them to appropriate support strategies. While Large Language Models (LLMs) excel in text generating, they fundamentally operate as probabilistic models trained on extensive datasets, lacking a true understanding of human thought processes and intentions. To address this limitation, we introduce the Intention Centric Chain-of-Thought (ICECoT) mechanism. ICECoT enables LLMs to mimic human reasoning by analyzing emotional states, inferring intentions, and selecting suitable support strategies, thereby generating more effective emotional support responses. To train the model with ICECoT and integrate expert knowledge, we design an automated annotation pipeline that produces high-quality training data. Furthermore, we develop a comprehensive evaluation scheme to assess emotional support efficacy and conduct extensive experiments to validate our framework. Our data and code are available at this https URL. 

---
# DynamicMind: A Tri-Mode Thinking System for Large Language Models 

**Authors**: Wei Li, Yanbin Wei, Qiushi Huang, Jiangyue Yan, Yang Chen, James T. Kwok, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05936)  

**Abstract**: Modern large language models (LLMs) often struggle to dynamically adapt their reasoning depth to varying task complexities, leading to suboptimal performance or inefficient resource utilization. To address this, we introduce DynamicMind, a novel tri-mode thinking system. DynamicMind empowers LLMs to autonomously select between Fast, Normal, and Slow thinking modes for zero-shot question answering (ZSQA) tasks through cognitive-inspired prompt engineering. Our framework's core innovations include: (1) expanding the established dual-process framework of fast and slow thinking into a tri-mode thinking system involving a normal thinking mode to preserve the intrinsic capabilities of LLM; (2) proposing the Thinking Density metric, which aligns computational resource allocation with problem complexity; and (3) developing the Thinking Mode Capacity (TMC) dataset and a lightweight Mind Router to predict the optimal thinking mode. Extensive experiments across diverse mathematical, commonsense, and scientific QA benchmarks demonstrate that DynamicMind achieves superior ZSQA capabilities while establishing an effective trade-off between performance and computational efficiency. 

---
# MoA: Heterogeneous Mixture of Adapters for Parameter-Efficient Fine-Tuning of Large Language Models 

**Authors**: Jie Cao, Tianwei Lin, Hongyang He, Rolan Yan, Wenqiao Zhang, Juncheng Li, Dongping Zhang, Siliang Tang, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05928)  

**Abstract**: Recent studies integrate Low-Rank Adaptation (LoRA) and Mixture-of-Experts (MoE) to further enhance the performance of parameter-efficient fine-tuning (PEFT) methods in Large Language Model (LLM) applications. Existing methods employ \emph{homogeneous} MoE-LoRA architectures composed of LoRA experts with either similar or identical structures and capacities. However, these approaches often suffer from representation collapse and expert load imbalance, which negatively impact the potential of LLMs. To address these challenges, we propose a \emph{heterogeneous} \textbf{Mixture-of-Adapters (MoA)} approach. This method dynamically integrates PEFT adapter experts with diverse structures, leveraging their complementary representational capabilities to foster expert specialization, thereby enhancing the effective transfer of pre-trained knowledge to downstream tasks. MoA supports two variants: \textbf{(i)} \textit{Soft MoA} achieves fine-grained integration by performing a weighted fusion of all expert outputs; \textbf{(ii)} \textit{Sparse MoA} activates adapter experts sparsely based on their contribution, achieving this with negligible performance degradation. Experimental results demonstrate that heterogeneous MoA outperforms homogeneous MoE-LoRA methods in both performance and parameter efficiency. Our project is available at this https URL. 

---
# LengClaro2023: A Dataset of Administrative Texts in Spanish with Plain Language adaptations 

**Authors**: Belén Agüera-Marco, Itziar Gonzalez-Dios  

**Link**: [PDF](https://arxiv.org/pdf/2506.05927)  

**Abstract**: In this work, we present LengClaro2023, a dataset of legal-administrative texts in Spanish. Based on the most frequently used procedures from the Spanish Social Security website, we have created for each text two simplified equivalents. The first version follows the recommendations provided by arText claro. The second version incorporates additional recommendations from plain language guidelines to explore further potential improvements in the system. The linguistic resource created in this work can be used for evaluating automatic text simplification (ATS) systems in Spanish. 

---
# Generating Grounded Responses to Counter Misinformation via Learning Efficient Fine-Grained Critiques 

**Authors**: Xiaofei Xu, Xiuzhen Zhang, Ke Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.05924)  

**Abstract**: Fake news and misinformation poses a significant threat to society, making efficient mitigation essential. However, manual fact-checking is costly and lacks scalability. Large Language Models (LLMs) offer promise in automating counter-response generation to mitigate misinformation, but a critical challenge lies in their tendency to hallucinate non-factual information. Existing models mainly rely on LLM self-feedback to reduce hallucination, but this approach is computationally expensive. In this paper, we propose MisMitiFact, Misinformation Mitigation grounded in Facts, an efficient framework for generating fact-grounded counter-responses at scale. MisMitiFact generates simple critique feedback to refine LLM outputs, ensuring responses are grounded in evidence. We develop lightweight, fine-grained critique models trained on data sourced from readily available fact-checking sites to identify and correct errors in key elements such as numerals, entities, and topics in LLM generations. Experiments show that MisMitiFact generates counter-responses of comparable quality to LLMs' self-feedback while using significantly smaller critique models. Importantly, it achieves ~5x increase in feedback generation throughput, making it highly suitable for cost-effective, large-scale misinformation mitigation. Code and LLM prompt templates are at this https URL. 

---
# Route-and-Reason: Scaling Large Language Model Reasoning with Reinforced Model Router 

**Authors**: Chenyang Shao, Xinyang Liu, Yutang Lin, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05901)  

**Abstract**: Multi-step reasoning has proven essential for enhancing the problem-solving capabilities of Large Language Models (LLMs) by decomposing complex tasks into intermediate steps, either explicitly or implicitly. Extending the reasoning chain at test time through deeper thought processes or broader exploration, can furthur improve performance, but often incurs substantial costs due to the explosion in token usage. Yet, many reasoning steps are relatively simple and can be handled by more efficient smaller-scale language models (SLMs). This motivates hybrid approaches that allocate subtasks across models of varying capacities. However, realizing such collaboration requires accurate task decomposition and difficulty-aware subtask allocation, which is challenging. To address this, we propose R2-Reasoner, a novel framework that enables collaborative reasoning across heterogeneous LLMs by dynamically routing sub-tasks based on estimated complexity. At the core of our framework is a Reinforced Model Router, composed of a task decomposer and a subtask allocator. The task decomposer segments complex input queries into logically ordered subtasks, while the subtask allocator assigns each subtask to the most appropriate model, ranging from lightweight SLMs to powerful LLMs, balancing accuracy and efficiency. To train this router, we introduce a staged pipeline that combines supervised fine-tuning on task-specific datasets with Group Relative Policy Optimization algorithm, enabling self-supervised refinement through iterative reinforcement learning. Extensive experiments across four challenging benchmarks demonstrate that R2-Reasoner reduces API costs by 86.85% while maintaining or surpassing baseline accuracy. Our framework paves the way for more cost-effective and adaptive LLM reasoning. The code is open-source at this https URL . 

---
# Cross-lingual Collapse: How Language-Centric Foundation Models Shape Reasoning in Large Language Models 

**Authors**: Cheonbok Park, Jeonghoon Kim, Joosung Lee, Sanghwan Bae, Jaegul Choo, Kangmin Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2506.05850)  

**Abstract**: We identify \textbf{Cross-lingual Collapse}, a systematic drift in which the chain-of-thought (CoT) of a multilingual language model reverts to its dominant pre-training language even when the prompt is expressed in a different language. Recent large language models (LLMs) with reinforcement learning with verifiable reward (RLVR) have achieved strong logical reasoning performances by exposing their intermediate reasoning traces, giving rise to large reasoning models (LRMs). However, the mechanism behind multilingual reasoning in LRMs is not yet fully explored. To investigate the issue, we fine-tune multilingual LRMs with Group-Relative Policy Optimization (GRPO) on translated versions of the GSM$8$K and SimpleRL-Zoo datasets in three different languages: Chinese, Korean, and Ukrainian. During training, we monitor both task accuracy and language consistency of the reasoning chains. Our experiments reveal three key findings: (i) GRPO rapidly amplifies pre-training language imbalances, leading to the erosion of low-resource languages within just a few hundred updates; (ii) language consistency reward mitigates this drift but does so at the expense of an almost 5 - 10 pp drop in accuracy. and (iii) the resulting language collapse is severely damaging and largely irreversible, as subsequent fine-tuning struggles to steer the model back toward its original target-language reasoning capabilities. Together, these findings point to a remarkable conclusion: \textit{not all languages are trained equally for reasoning}. Furthermore, our paper sheds light on the roles of reward shaping, data difficulty, and pre-training priors in eliciting multilingual reasoning. 

---
# FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging 

**Authors**: Zichen Tang, Haihong E, Ziyan Ma, Haoyang He, Jiacheng Liu, Zhongjun Yang, Zihua Rong, Rongjin Li, Kun Ji, Qing Huang, Xinyang Hu, Yang Liu, Qianhe Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.05828)  

**Abstract**: We introduce FinanceReasoning, a novel benchmark designed to evaluate the reasoning capabilities of large reasoning models (LRMs) in financial numerical reasoning problems. Compared to existing benchmarks, our work provides three key advancements. (1) Credibility: We update 15.6% of the questions from four public datasets, annotating 908 new questions with detailed Python solutions and rigorously refining evaluation standards. This enables an accurate assessment of the reasoning improvements of LRMs. (2) Comprehensiveness: FinanceReasoning covers 67.8% of financial concepts and formulas, significantly surpassing existing datasets. Additionally, we construct 3,133 Python-formatted functions, which enhances LRMs' financial reasoning capabilities through refined knowledge (e.g., 83.2% $\rightarrow$ 91.6% for GPT-4o). (3) Challenge: Models are required to apply multiple financial formulas for precise numerical reasoning on 238 Hard problems. The best-performing model (i.e., OpenAI o1 with PoT) achieves 89.1% accuracy, yet LRMs still face challenges in numerical precision. We demonstrate that combining Reasoner and Programmer models can effectively enhance LRMs' performance (e.g., 83.2% $\rightarrow$ 87.8% for DeepSeek-R1). Our work paves the way for future research on evaluating and improving LRMs in domain-specific complex reasoning tasks. 

---
# MAPLE: Multi-Agent Adaptive Planning with Long-Term Memory for Table Reasoning 

**Authors**: Ye Bai, Minghan Wang, Thuy-Trang Vu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05813)  

**Abstract**: Table-based question answering requires complex reasoning capabilities that current LLMs struggle to achieve with single-pass inference. Existing approaches, such as Chain-of-Thought reasoning and question decomposition, lack error detection mechanisms and discard problem-solving experiences, contrasting sharply with how humans tackle such problems. In this paper, we propose MAPLE (Multi-agent Adaptive Planning with Long-term mEmory), a novel framework that mimics human problem-solving through specialized cognitive agents working in a feedback-driven loop. MAPLE integrates 4 key components: (1) a Solver using the ReAct paradigm for reasoning, (2) a Checker for answer verification, (3) a Reflector for error diagnosis and strategy correction, and (4) an Archiver managing long-term memory for experience reuse and evolution. Experiments on WiKiTQ and TabFact demonstrate significant improvements over existing methods, achieving state-of-the-art performance across multiple LLM backbones. 

---
# Discrete Minds in a Continuous World: Do Language Models Know Time Passes? 

**Authors**: Minghan Wang, Ye Bai, Thuy-Trang Vu, Ehsan Shareghi, Gholamreza Haffari  

**Link**: [PDF](https://arxiv.org/pdf/2506.05790)  

**Abstract**: While Large Language Models (LLMs) excel at temporal reasoning tasks like event ordering and duration estimation, their ability to perceive the actual passage of time remains unexplored. We investigate whether LLMs perceive the passage of time and adapt their decision-making accordingly through three complementary experiments. First, we introduce the Token-Time Hypothesis, positing that LLMs can map discrete token counts to continuous wall-clock time, and validate this through a dialogue duration judgment task. Second, we demonstrate that LLMs could use this awareness to adapt their response length while maintaining accuracy when users express urgency in question answering tasks. Finally, we develop BombRush, an interactive navigation challenge that examines how LLMs modify behavior under progressive time pressure in dynamic environments. Our findings indicate that LLMs possess certain awareness of time passage, enabling them to bridge discrete linguistic tokens and continuous physical time, though this capability varies with model size and reasoning abilities. This work establishes a theoretical foundation for enhancing temporal awareness in LLMs for time-sensitive applications. 

---
# dots.llm1 Technical Report 

**Authors**: Bi Huo, Bin Tu, Cheng Qin, Da Zheng, Debing Zhang, Dongjie Zhang, En Li, Fu Guo, Jian Yao, Jie Lou, Junfeng Tian, Li Hu, Ran Zhu, Shengdong Chen, Shuo Liu, Su Guang, Te Wo, Weijun Zhang, Xiaoming Shi, Xinxin Peng, Xing Wu, Yawen Liu, Yuqiu Ji, Ze Wen, Zhenhai Liu, Zichao Li, Zilong Liao  

**Link**: [PDF](https://arxiv.org/pdf/2506.05767)  

**Abstract**: Mixture of Experts (MoE) models have emerged as a promising paradigm for scaling language models efficiently by activating only a subset of parameters for each input token. In this report, we present dots.llm1, a large-scale MoE model that activates 14B parameters out of a total of 142B parameters, delivering performance on par with state-of-the-art models while reducing training and inference costs. Leveraging our meticulously crafted and efficient data processing pipeline, dots.llm1 achieves performance comparable to Qwen2.5-72B after pretraining on 11.2T high-quality tokens and post-training to fully unlock its capabilities. Notably, no synthetic data is used during pretraining. To foster further research, we open-source intermediate training checkpoints at every one trillion tokens, providing valuable insights into the learning dynamics of large language models. 

---
# BioMol-MQA: A Multi-Modal Question Answering Dataset For LLM Reasoning Over Bio-Molecular Interactions 

**Authors**: Saptarshi Sengupta, Shuhua Yang, Paul Kwong Yu, Fali Wang, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05766)  

**Abstract**: Retrieval augmented generation (RAG) has shown great power in improving Large Language Models (LLMs). However, most existing RAG-based LLMs are dedicated to retrieving single modality information, mainly text; while for many real-world problems, such as healthcare, information relevant to queries can manifest in various modalities such as knowledge graph, text (clinical notes), and complex molecular structure. Thus, being able to retrieve relevant multi-modality domain-specific information, and reason and synthesize diverse knowledge to generate an accurate response is important. To address the gap, we present BioMol-MQA, a new question-answering (QA) dataset on polypharmacy, which is composed of two parts (i) a multimodal knowledge graph (KG) with text and molecular structure for information retrieval; and (ii) challenging questions that designed to test LLM capabilities in retrieving and reasoning over multimodal KG to answer questions. Our benchmarks indicate that existing LLMs struggle to answer these questions and do well only when given the necessary background data, signaling the necessity for strong RAG frameworks. 

---
# Writing-RL: Advancing Long-form Writing via Adaptive Curriculum Reinforcement Learning 

**Authors**: Xuanyu Lei, Chenliang Li, Yuning Wu, Kaiming Liu, Weizhou Shen, Peng Li, Ming Yan, Ji Zhang, Fei Huang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05760)  

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled strong performance in long-form writing, yet existing supervised fine-tuning (SFT) approaches suffer from limitations such as data saturation and restricted learning capacity bounded by teacher signals. In this work, we present Writing-RL: an Adaptive Curriculum Reinforcement Learning framework to advance long-form writing capabilities beyond SFT. The framework consists of three key components: Margin-aware Data Selection strategy that prioritizes samples with high learning potential, Pairwise Comparison Reward mechanism that provides discriminative learning signals in the absence of verifiable rewards, and Dynamic Reference Scheduling approach, which plays a particularly critical role by adaptively adjusting task difficulty based on evolving model performance. Experiments on 7B-scale writer models show that our RL framework largely improves long-form writing performance over strong SFT baselines. Furthermore, we observe that models trained with long-output RL generalize surprisingly well to long-input reasoning tasks, potentially offering a promising perspective for rethinking long-context training. 

---
# LLM-Symbolic Integration for Robust Temporal Tabular Reasoning 

**Authors**: Atharv Kulkarni, Kushagra Dixit, Vivek Srikumar, Dan Roth, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.05746)  

**Abstract**: Temporal tabular question answering presents a significant challenge for Large Language Models (LLMs), requiring robust reasoning over structured data, which is a task where traditional prompting methods often fall short. These methods face challenges such as memorization, sensitivity to table size, and reduced performance on complex queries. To overcome these limitations, we introduce TempTabQA-C, a synthetic dataset designed for systematic and controlled evaluations, alongside a symbolic intermediate representation that transforms tables into database schemas. This structured approach allows LLMs to generate and execute SQL queries, enhancing generalization and mitigating biases. By incorporating adaptive few-shot prompting with contextually tailored examples, our method achieves superior robustness, scalability, and performance. Experimental results consistently highlight improvements across key challenges, setting a new benchmark for robust temporal reasoning with LLMs. 

---
# Do LLMs Really Forget? Evaluating Unlearning with Knowledge Correlation and Confidence Awareness 

**Authors**: Rongzhe Wei, Peizhi Niu, Hans Hao-Hsun Hsu, Ruihan Wu, Haoteng Yin, Mohsen Ghassemi, Yifan Li, Vamsi K. Potluru, Eli Chien, Kamalika Chaudhuri, Olgica Milenkovic, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05735)  

**Abstract**: Machine unlearning techniques aim to mitigate unintended memorization in large language models (LLMs). However, existing approaches predominantly focus on the explicit removal of isolated facts, often overlooking latent inferential dependencies and the non-deterministic nature of knowledge within LLMs. Consequently, facts presumed forgotten may persist implicitly through correlated information. To address these challenges, we propose a knowledge unlearning evaluation framework that more accurately captures the implicit structure of real-world knowledge by representing relevant factual contexts as knowledge graphs with associated confidence scores. We further develop an inference-based evaluation protocol leveraging powerful LLMs as judges; these judges reason over the extracted knowledge subgraph to determine unlearning success. Our LLM judges utilize carefully designed prompts and are calibrated against human evaluations to ensure their trustworthiness and stability. Extensive experiments on our newly constructed benchmark demonstrate that our framework provides a more realistic and rigorous assessment of unlearning performance. Moreover, our findings reveal that current evaluation strategies tend to overestimate unlearning effectiveness. Our code is publicly available at this https URL. 

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
# Being Strong Progressively! Enhancing Knowledge Distillation of Large Language Models through a Curriculum Learning Framework 

**Authors**: Lingyuan Liu, Mengxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05695)  

**Abstract**: Knowledge Distillation (KD) compresses large language models (LLMs) by transferring the teacher model's capabilities to a smaller student model, reducing inference cost and memory usage while maintaining performance. However, existing KD methods for LLMs often fail to prevent significant shifts in the student model's distribution during training, leading to issues such as catastrophic forgetting, mode collapse, and training-inference mismatch. To address these challenges, we propose a novel, plug-in curriculum learning framework inspired by the strength training principle of "progressive overload" (POCL), which can be seamlessly integrated into existing white-box KD approaches with minimal computational overhead. The framework comprises two core components: (1) a difficulty measurer that ranks and partitions training samples from easy to hard, and (2) a training scheduler that incrementally introduces these subsets into the distillation process at fixed intervals while applying loss functions with progressively rising temperatures. By starting with the easiest samples and progressively increasing the difficulty, the approach enhances both the stability and efficiency of learning. Extensive experiments in instruction-following settings demonstrate that POCL consistently improves the performance of distilled student models across various white-box KD methods and model families. Our findings highlight the effectiveness of sorted training samples in KD for LLMs. More generally, our work demonstrates how to structure training data within the KD process to enhance the stability and performance of distilled LLMs. 

---
# When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation 

**Authors**: Zhishang Xiang, Chuanjie Wu, Qinggang Zhang, Shengyuan Chen, Zijin Hong, Xiao Huang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.05690)  

**Abstract**: Graph retrieval-augmented generation (GraphRAG) has emerged as a powerful paradigm for enhancing large language models (LLMs) with external knowledge. It leverages graphs to model the hierarchical structure between specific concepts, enabling more coherent and effective knowledge retrieval for accurate this http URL its conceptual promise, recent studies report that GraphRAG frequently underperforms vanilla RAG on many real-world tasks. This raises a critical question: Is GraphRAG really effective, and in which scenarios do graph structures provide measurable benefits for RAG systems? To address this, we propose GraphRAG-Bench, a comprehensive benchmark designed to evaluate GraphRAG models onboth hierarchical knowledge retrieval and deep contextual reasoning. GraphRAG-Bench features a comprehensive dataset with tasks of increasing difficulty, coveringfact retrieval, complex reasoning, contextual summarization, and creative generation, and a systematic evaluation across the entire pipeline, from graph constructionand knowledge retrieval to final generation. Leveraging this novel benchmark, we systematically investigate the conditions when GraphRAG surpasses traditional RAG and the underlying reasons for its success, offering guidelines for its practical application. All related resources and analyses are collected for the community at this https URL. 

---
# A Unified Representation for Continuity and Discontinuity: Syntactic and Computational Motivations 

**Authors**: Ratna Kandala, Prakash Mondal  

**Link**: [PDF](https://arxiv.org/pdf/2506.05686)  

**Abstract**: This paper advances a unified representation of linguistic structure for three grammar formalisms, namely, Phrase Structure Grammar (PSG), Dependency Grammar (DG) and Categorial Grammar (CG) from the perspective of syntactic and computational complexity considerations. The correspondence principle is proposed to enable a unified representation of the representational principles from PSG, DG, and CG. To that end, the paper first illustrates a series of steps in achieving a unified representation for a discontinuous subordinate clause from Turkish as an illustrative case. This affords a new way of approaching discontinuity in natural language from a theoretical point of view that unites and integrates the basic tenets of PSG, DG, and CG, with significant consequences for syntactic analysis. Then this paper demonstrates that a unified representation can simplify computational complexity with regards to the neurocognitive representation and processing of both continuous and discontinuous sentences vis-à-vis the basic principles of PSG, DG, and CG. 

---
# Zero-Shot Event Causality Identification via Multi-source Evidence Fuzzy Aggregation with Large Language Models 

**Authors**: Zefan Zeng, Xingchen Hu, Qing Cheng, Weiping Ding, Wentao Li, Zhong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05675)  

**Abstract**: Event Causality Identification (ECI) aims to detect causal relationships between events in textual contexts. Existing ECI models predominantly rely on supervised methodologies, suffering from dependence on large-scale annotated data. Although Large Language Models (LLMs) enable zero-shot ECI, they are prone to causal hallucination-erroneously establishing spurious causal links. To address these challenges, we propose MEFA, a novel zero-shot framework based on Multi-source Evidence Fuzzy Aggregation. First, we decompose causality reasoning into three main tasks (temporality determination, necessity analysis, and sufficiency verification) complemented by three auxiliary tasks. Second, leveraging meticulously designed prompts, we guide LLMs to generate uncertain responses and deterministic outputs. Finally, we quantify LLM's responses of sub-tasks and employ fuzzy aggregation to integrate these evidence for causality scoring and causality determination. Extensive experiments on three benchmarks demonstrate that MEFA outperforms second-best unsupervised baselines by 6.2% in F1-score and 9.3% in precision, while significantly reducing hallucination-induced errors. In-depth analysis verify the effectiveness of task decomposition and the superiority of fuzzy aggregation. 

---
# Can LLMs Express Personality Across Cultures? Introducing CulturalPersonas for Evaluating Trait Alignment 

**Authors**: Priyanka Dey, Yugal Khanter, Aayush Bothra, Jieyu Zhao, Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2506.05670)  

**Abstract**: As LLMs become central to interactive applications, ranging from tutoring to mental health, the ability to express personality in culturally appropriate ways is increasingly important. While recent works have explored personality evaluation of LLMs, they largely overlook the interplay between culture and personality. To address this, we introduce CulturalPersonas, the first large-scale benchmark with human validation for evaluating LLMs' personality expression in culturally grounded, behaviorally rich contexts. Our dataset spans 3,000 scenario-based questions across six diverse countries, designed to elicit personality through everyday scenarios rooted in local values. We evaluate three LLMs, using both multiple-choice and open-ended response formats. Our results show that CulturalPersonas improves alignment with country-specific human personality distributions (over a 20% reduction in Wasserstein distance across models and countries) and elicits more expressive, culturally coherent outputs compared to existing benchmarks. CulturalPersonas surfaces meaningful modulated trait outputs in response to culturally grounded prompts, offering new directions for aligning LLMs to global norms of behavior. By bridging personality expression and cultural nuance, we envision that CulturalPersonas will pave the way for more socially intelligent and globally adaptive LLMs. 

---
# A Fictional Q&A Dataset for Studying Memorization and Knowledge Acquisition 

**Authors**: John Kirchenbauer, Janny Mongkolsupawan, Yuxin Wen, Tom Goldstein, Daphne Ippolito  

**Link**: [PDF](https://arxiv.org/pdf/2506.05639)  

**Abstract**: When language models are trained on textual data, they acquire both knowledge about the structure of language as well as knowledge of facts about the world. At inference time, their knowledge of facts can be leveraged to solve interesting problems and perform useful knowledge work for users. It is well known that language models can verbatim memorize long sequences from their training data. However, it is much less well understood how language models memorize facts seen during training. In this work, we propose a new dataset to specifically empower researchers to study the dual processes of fact memorization and verbatim sequence memorization. The dataset consists of synthetically-generated, webtext-like documents about fictional events, as well as question-answer pairs about the events. We conduct training experiments showing how synthetic data about fictional events can be effective in teasing apart different forms of memorization. We also document the challenges in effectively building realistic, fictional synthetic data. 

---
# IYKYK: Using language models to decode extremist cryptolects 

**Authors**: Christine de Kock, Arij Riabi, Zeerak Talat, Michael Sejr Schlichtkrull, Pranava Madhyastha, Ed Hovy  

**Link**: [PDF](https://arxiv.org/pdf/2506.05635)  

**Abstract**: Extremist groups develop complex in-group language, also referred to as cryptolects, to exclude or mislead outsiders. We investigate the ability of current language technologies to detect and interpret the cryptolects of two online extremist platforms. Evaluating eight models across six tasks, our results indicate that general purpose LLMs cannot consistently detect or decode extremist language. However, performance can be significantly improved by domain adaptation and specialised prompting techniques. These results provide important insights to inform the development and deployment of automated moderation technologies. We further develop and release novel labelled and unlabelled datasets, including 19.4M posts from extremist platforms and lexicons validated by human experts. 

---
# Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs 

**Authors**: Ananth Muppidi, Abhilash Nandy, Sambaran Bandyopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2506.05629)  

**Abstract**: The performance of large language models in domain-specific tasks necessitates fine-tuning, which is computationally expensive and technically challenging. This paper focuses on parameter-efficient fine-tuning using soft prompting, a promising approach that adapts pre-trained models to downstream tasks by learning a small set of parameters. We propose a novel Input Dependent Soft Prompting technique with a self-Attention Mechanism (ID-SPAM) that generates soft prompts based on the input tokens and attends different tokens with varying importance. Our method is simple and efficient, keeping the number of trainable parameters small. We show the merits of the proposed approach compared to state-of-the-art techniques on various tasks and show the improved zero shot domain transfer capability. 

---
# Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking 

**Authors**: Zhecheng Sheng, Xiruo Ding, Brian Hur, Changye Li, Trevor Cohen, Serguei Pakhomov  

**Link**: [PDF](https://arxiv.org/pdf/2506.05610)  

**Abstract**: Deep transformer models have been used to detect linguistic anomalies in patient transcripts for early Alzheimer's disease (AD) screening. While pre-trained neural language models (LMs) fine-tuned on AD transcripts perform well, little research has explored the effects of the gender of the speakers represented by these transcripts. This work addresses gender confounding in dementia detection and proposes two methods: the $\textit{Extended Confounding Filter}$ and the $\textit{Dual Filter}$, which isolate and ablate weights associated with gender. We evaluate these methods on dementia datasets with first-person narratives from patients with cognitive impairment and healthy controls. Our results show transformer models tend to overfit to training data distributions. Disrupting gender-related weights results in a deconfounded dementia classifier, with the trade-off of slightly reduced dementia detection performance. 

---
# OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation 

**Authors**: Ziyi Wang, Yuxuan Lu, Wenbo Li, Amirali Amini, Bo Sun, Yakov Bart, Weimin Lyu, Jiri Gesi, Tian Wang, Jing Huang, Yu Su, Upol Ehsan, Malihe Alikhani, Toby Jia-Jun Li, Lydia Chilton, Dakuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05606)  

**Abstract**: Can large language models (LLMs) accurately simulate the next web action of a specific user? While LLMs have shown promising capabilities in generating ``believable'' human behaviors, evaluating their ability to mimic real user behaviors remains an open challenge, largely due to the lack of high-quality, publicly available datasets that capture both the observable actions and the internal reasoning of an actual human user. To address this gap, we introduce OPERA, a novel dataset of Observation, Persona, Rationale, and Action collected from real human participants during online shopping sessions. OPERA is the first public dataset that comprehensively captures: user personas, browser observations, fine-grained web actions, and self-reported just-in-time rationales. We developed both an online questionnaire and a custom browser plugin to gather this dataset with high fidelity. Using OPERA, we establish the first benchmark to evaluate how well current LLMs can predict a specific user's next action and rationale with a given persona and <observation, action, rationale> history. This dataset lays the groundwork for future research into LLM agents that aim to act as personalized digital twins for human. 

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
# Combating Misinformation in the Arab World: Challenges & Opportunities 

**Authors**: Azza Abouzied, Firoj Alam, Raian Ali, Paolo Papotti  

**Link**: [PDF](https://arxiv.org/pdf/2506.05582)  

**Abstract**: Misinformation and disinformation pose significant risks globally, with the Arab region facing unique vulnerabilities due to geopolitical instabilities, linguistic diversity, and cultural nuances. We explore these challenges through the key facets of combating misinformation: detection, tracking, mitigation and community-engagement. We shed light on how connecting with grass-roots fact-checking organizations, understanding cultural norms, promoting social correction, and creating strong collaborative information networks can create opportunities for a more resilient information ecosystem in the Arab world. 

---
# Improving LLMs with a knowledge from databases 

**Authors**: Petr Máša  

**Link**: [PDF](https://arxiv.org/pdf/2506.05560)  

**Abstract**: Large language models (LLMs) are achieving significant progress almost every moment now. Many advanced techniques have been introduced and widely accepted, like retrieval-augmentation generation (RAG), agents, and tools. Tools can query the database to answer questions from structured data files or perform groupings or other statistics. This unlocks huge opportunities, such as it can answer any question, but also poses threats, such as safety, because there is no control over the commands that are created. We would like to discuss whether we can create a new method that improves answers based on dataset/database via some interpretable ML methods, namely enhanced association rules. The advantage would be if the method can be also used in some safe technique like RAG. Association rules have a sound history. Since the introduction of CN2 and aproiri, many enhancements have been made. In parallel, enhanced association rules have been introduced and evolved over the last 40 years. The general problem is typically that there are too many rules. There are some techniques for handling it, but when LLM emerged, it turned out to be the best use case for the RAG technique for LLMs. We proposed a method that generates a ruleset based on defined knowledge patterns, then converts rules into text form via a rule-to-text converter, and includes the result as an RAG into LLM. We compared this method with ChatGPT (even with using agents) and we have discovered a significant improvement in answering questions based on the dataset. We have also tried several strategies how much rules to generate. We found this improvement interesting. Moreover, it can also be improved in many ways as future work, like incorporating other patterns, the use of rule mining as an agent, and many others. 

---
# Multidimensional Analysis of Specific Language Impairment Using Unsupervised Learning Through PCA and Clustering 

**Authors**: Niruthiha Selvanayagam  

**Link**: [PDF](https://arxiv.org/pdf/2506.05498)  

**Abstract**: Specific Language Impairment (SLI) affects approximately 7 percent of children, presenting as isolated language deficits despite normal cognitive abilities, sensory systems, and supportive environments. Traditional diagnostic approaches often rely on standardized assessments, which may overlook subtle developmental patterns. This study aims to identify natural language development trajectories in children with and without SLI using unsupervised machine learning techniques, providing insights for early identification and targeted interventions. Narrative samples from 1,163 children aged 4-16 years across three corpora (Conti-Ramsden 4, ENNI, and Gillam) were analyzed using Principal Component Analysis (PCA) and clustering. A total of 64 linguistic features were evaluated to uncover developmental trajectories and distinguish linguistic profiles. Two primary clusters emerged: (1) high language production with low SLI prevalence, and (2) limited production but higher syntactic complexity with higher SLI prevalence. Additionally, boundary cases exhibited intermediate traits, supporting a continuum model of language abilities. Findings suggest SLI manifests primarily through reduced production capacity rather than syntactic complexity deficits. The results challenge categorical diagnostic frameworks and highlight the potential of unsupervised learning techniques for refining diagnostic criteria and intervention strategies. 

---
# MLLM-CL: Continual Learning for Multimodal Large Language Models 

**Authors**: Hongbo Zhao, Fei Zhu, Rundong Wang, Gaofeng Meng, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05453)  

**Abstract**: Recent Multimodal Large Language Models (MLLMs) excel in vision-language understanding but face challenges in adapting to dynamic real-world scenarios that require continuous integration of new knowledge and skills. While continual learning (CL) offers a potential solution, existing benchmarks and methods suffer from critical limitations. In this paper, we introduce MLLM-CL, a novel benchmark encompassing domain and ability continual learning, where the former focuses on independently and identically distributed (IID) evaluation across evolving mainstream domains, whereas the latter evaluates on non-IID scenarios with emerging model ability. Methodologically, we propose preventing catastrophic interference through parameter isolation, along with an MLLM-based routing mechanism. Extensive experiments demonstrate that our approach can integrate domain-specific knowledge and functional abilities with minimal forgetting, significantly outperforming existing methods. 

---
# Automatically Detecting Amusing Games in Wordle 

**Authors**: Ronaldo Luo, Gary Liang, Cindy Liu, Adam Kabbara, Minahil Bakhtawar, Kina Kim, Michael Guerzhoy  

**Link**: [PDF](https://arxiv.org/pdf/2506.05415)  

**Abstract**: We explore automatically predicting which Wordle games Reddit users find amusing.
We scrape approximately 80k reactions by Reddit users to Wordle games from Reddit, classify the reactions as expressing amusement or not using OpenAI's GPT-3.5 using few-shot prompting, and verify that GPT-3.5's labels roughly correspond to human labels.
We then extract features from Wordle games that can predict user amusement. We demonstrate that the features indeed provide a (weak) signal that predicts user amusement as predicted by GPT-3.5.
Our results indicate that user amusement at Wordle games can be predicted computationally to some extent. We explore which features of the game contribute to user amusement.
We find that user amusement is predictable, indicating a measurable aspect of creativity infused into Wordle games through humor. 

---
# SmoothRot: Combining Channel-Wise Scaling and Rotation for Quantization-Friendly LLMs 

**Authors**: Patrik Czakó, Gábor Kertész, Sándor Szénási  

**Link**: [PDF](https://arxiv.org/pdf/2506.05413)  

**Abstract**: We present SmoothRot, a novel post-training quantization technique to enhance the efficiency of 4-bit quantization in Large Language Models (LLMs). SmoothRot addresses the critical challenge of massive activation outliers, by integrating channel-wise scaling with Hadamard transformations. Our technique effectively transforms extreme outliers into quantization-friendly activations, significantly improving quantization accuracy. Experiments conducted on popular LLMs (LLaMA2 7B, LLaMA3.1 8B, and Mistral 7B) demonstrate that SmoothRot consistently reduces the performance gap between quantized and FP16 models by approximately 10-30\% across language generation and zero-shot reasoning tasks, without introducing additional inference latency. Code is available at this https URL. 

---
# Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry for Long-Context LLMs 

**Authors**: Wanyun Cui, Mingwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05410)  

**Abstract**: Recent advances in Large Language Models (LLMs) have highlighted the critical importance of extending context length, yet the quadratic complexity of attention mechanisms poses significant challenges for efficient long-context modeling. KV cache compression has emerged as a key approach to address this challenge. Through extensive empirical analysis, we reveal a fundamental yet previously overlooked asymmetry in KV caches: while adjacent keys receive similar attention weights (local homogeneity), adjacent values demonstrate distinct heterogeneous distributions. This key-value asymmetry reveals a critical limitation in existing compression methods that treat keys and values uniformly. To address the limitation, we propose a training-free compression framework (AsymKV) that combines homogeneity-based key merging with a mathematically proven lossless value compression. Extensive experiments demonstrate that AsymKV consistently outperforms existing long-context methods across various tasks and base models. For example, on LLaMA3.1-8B, AsymKV achieves an average score of 43.95 on LongBench, surpassing SOTA methods like H$_2$O (38.89) by a large margin. 

---
# Auto Review: Second Stage Error Detection for Highly Accurate Information Extraction from Phone Conversations 

**Authors**: Ayesha Qamar, Arushi Raghuvanshi, Conal Sathi, Youngseo Son  

**Link**: [PDF](https://arxiv.org/pdf/2506.05400)  

**Abstract**: Automating benefit verification phone calls saves time in healthcare and helps patients receive treatment faster. It is critical to obtain highly accurate information in these phone calls, as it can affect a patient's healthcare journey. Given the noise in phone call transcripts, we have a two-stage system that involves a post-call review phase for potentially noisy fields, where human reviewers manually verify the extracted data$\unicode{x2013}$a labor-intensive task. To automate this stage, we introduce Auto Review, which significantly reduces manual effort while maintaining a high bar for accuracy. This system, being highly reliant on call transcripts, suffers a performance bottleneck due to automatic speech recognition (ASR) issues. This problem is further exacerbated by the use of domain-specific jargon in the calls. In this work, we propose a second-stage postprocessing pipeline for accurate information extraction. We improve accuracy by using multiple ASR alternatives and a pseudo-labeling approach that does not require manually corrected transcripts. Experiments with general-purpose large language models and feature-based model pipelines demonstrate substantial improvements in the quality of corrected call transcripts, thereby enhancing the efficiency of Auto Review. 

---
# Are Large Language Models Good Temporal Graph Learners? 

**Authors**: Shenyang Huang, Ali Parviz, Emma Kondrup, Zachary Yang, Zifeng Ding, Michael Bronstein, Reihaneh Rabbany, Guillaume Rabusseau  

**Link**: [PDF](https://arxiv.org/pdf/2506.05393)  

**Abstract**: Large Language Models (LLMs) have recently driven significant advancements in Natural Language Processing and various other applications. While a broad range of literature has explored the graph-reasoning capabilities of LLMs, including their use of predictors on graphs, the application of LLMs to dynamic graphs -- real world evolving networks -- remains relatively unexplored. Recent work studies synthetic temporal graphs generated by random graph models, but applying LLMs to real-world temporal graphs remains an open question. To address this gap, we introduce Temporal Graph Talker (TGTalker), a novel temporal graph learning framework designed for LLMs. TGTalker utilizes the recency bias in temporal graphs to extract relevant structural information, converted to natural language for LLMs, while leveraging temporal neighbors as additional information for prediction. TGTalker demonstrates competitive link prediction capabilities compared to existing Temporal Graph Neural Network (TGNN) models. Across five real-world networks, TGTalker performs competitively with state-of-the-art temporal graph methods while consistently outperforming popular models such as TGN and HTGN. Furthermore, TGTalker generates textual explanations for each prediction, thus opening up exciting new directions in explainability and interpretability for temporal link prediction. The code is publicly available at this https URL. 

---
# Understanding Gender Bias in AI-Generated Product Descriptions 

**Authors**: Markelle Kelly, Mohammad Tahaei, Padhraic Smyth, Lauren Wilcox  

**Link**: [PDF](https://arxiv.org/pdf/2506.05390)  

**Abstract**: While gender bias in large language models (LLMs) has been extensively studied in many domains, uses of LLMs in e-commerce remain largely unexamined and may reveal novel forms of algorithmic bias and harm. Our work investigates this space, developing data-driven taxonomic categories of gender bias in the context of product description generation, which we situate with respect to existing general purpose harms taxonomies. We illustrate how AI-generated product descriptions can uniquely surface gender biases in ways that require specialized detection and mitigation approaches. Further, we quantitatively analyze issues corresponding to our taxonomic categories in two models used for this task -- GPT-3.5 and an e-commerce-specific LLM -- demonstrating that these forms of bias commonly occur in practice. Our results illuminate unique, under-explored dimensions of gender bias, such as assumptions about clothing size, stereotypical bias in which features of a product are advertised, and differences in the use of persuasive language. These insights contribute to our understanding of three types of AI harms identified by current frameworks: exclusionary norms, stereotyping, and performance disparities, particularly for the context of e-commerce. 

---
# taz2024full: Analysing German Newspapers for Gender Bias and Discrimination across Decades 

**Authors**: Stefanie Urchs, Veronika Thurner, Matthias Aßenmacher, Christian Heumann, Stephanie Thiemichen  

**Link**: [PDF](https://arxiv.org/pdf/2506.05388)  

**Abstract**: Open-access corpora are essential for advancing natural language processing (NLP) and computational social science (CSS). However, large-scale resources for German remain limited, restricting research on linguistic trends and societal issues such as gender bias. We present taz2024full, the largest publicly available corpus of German newspaper articles to date, comprising over 1.8 million texts from taz, spanning 1980 to 2024.
As a demonstration of the corpus's utility for bias and discrimination research, we analyse gender representation across four decades of reporting. We find a consistent overrepresentation of men, but also a gradual shift toward more balanced coverage in recent years. Using a scalable, structured analysis pipeline, we provide a foundation for studying actor mentions, sentiment, and linguistic framing in German journalistic texts.
The corpus supports a wide range of applications, from diachronic language analysis to critical media studies, and is freely available to foster inclusive and reproducible research in German-language NLP. 

---
# Advancing Decoding Strategies: Enhancements in Locally Typical Sampling for LLMs 

**Authors**: Jaydip Sen, Saptarshi Sengupta. Subhasis Dasgupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.05387)  

**Abstract**: This chapter explores advancements in decoding strategies for large language models (LLMs), focusing on enhancing the Locally Typical Sampling (LTS) algorithm. Traditional decoding methods, such as top-k and nucleus sampling, often struggle to balance fluency, diversity, and coherence in text generation. To address these challenges, Adaptive Semantic-Aware Typicality Sampling (ASTS) is proposed as an improved version of LTS, incorporating dynamic entropy thresholding, multi-objective scoring, and reward-penalty adjustments. ASTS ensures contextually coherent and diverse text generation while maintaining computational efficiency. Its performance is evaluated across multiple benchmarks, including story generation and abstractive summarization, using metrics such as perplexity, MAUVE, and diversity scores. Experimental results demonstrate that ASTS outperforms existing sampling techniques by reducing repetition, enhancing semantic alignment, and improving fluency. 

---
# Beyond RAG: Reinforced Reasoning Augmented Generation for Clinical Notes 

**Authors**: Lo Pang-Yun Ting, Chengshuai Zhao, Yu-Hua Zeng, Yuan Jee Lim, Kun-Ta Chuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05386)  

**Abstract**: Clinical note generation aims to automatically produce free-text summaries of a patient's condition and diagnostic process, with discharge instructions being a representative long-form example. While recent large language model (LLM)-based methods pre-trained on general clinical corpora show promise in clinical text generation, they fall short in producing long-form notes from limited patient information. In this paper, we propose R2AG, the first reinforced retriever for long-form discharge instruction generation based on pre-admission data. R2AG is trained with reinforcement learning to retrieve reasoning paths from a medical knowledge graph, providing explicit semantic guidance to the LLM. To bridge the information gap, we propose Group-Based Retriever Optimization (GRO) which improves retrieval quality with group-relative rewards, encouraging reasoning leaps for deeper inference by the LLM. Comprehensive experiments on the MIMIC-IV-Note dataset show that R2AG outperforms baselines in both clinical efficacy and natural language generation metrics. Further analysis reveals that R2AG fills semantic gaps in sparse input scenarios, and retrieved reasoning paths help LLMs avoid clinical misinterpretation by focusing on key evidence and following coherent reasoning. 

---
# LLMs Can Also Do Well! Breaking Barriers in Semantic Role Labeling via Large Language Models 

**Authors**: Xinxin Li, Huiyao Chen, Chengjun Liu, Jing Li, Meishan Zhang, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05385)  

**Abstract**: Semantic role labeling (SRL) is a crucial task of natural language processing (NLP). Although generative decoder-based large language models (LLMs) have achieved remarkable success across various NLP tasks, they still lag behind state-of-the-art encoder-decoder (BERT-like) models in SRL. In this work, we seek to bridge this gap by equipping LLMs for SRL with two mechanisms: (a) retrieval-augmented generation and (b) self-correction. The first mechanism enables LLMs to leverage external linguistic knowledge such as predicate and argument structure descriptions, while the second allows LLMs to identify and correct inconsistent SRL outputs. We conduct extensive experiments on three widely-used benchmarks of SRL (CPB1.0, CoNLL-2009, and CoNLL-2012). Results demonstrate that our method achieves state-of-the-art performance in both Chinese and English, marking the first successful application of LLMs to surpass encoder-decoder approaches in SRL. 

---
# EvidenceOutcomes: a Dataset of Clinical Trial Publications with Clinically Meaningful Outcomes 

**Authors**: Yiliang Zhou, Abigail M. Newbury, Gongbo Zhang, Betina Ross Idnay, Hao Liu, Chunhua Weng, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.05380)  

**Abstract**: The fundamental process of evidence extraction and synthesis in evidence-based medicine involves extracting PICO (Population, Intervention, Comparison, and Outcome) elements from biomedical literature. However, Outcomes, being the most complex elements, are often neglected or oversimplified in existing benchmarks. To address this issue, we present EvidenceOutcomes, a novel, large, annotated corpus of clinically meaningful outcomes extracted from biomedical literature. We first developed a robust annotation guideline for extracting clinically meaningful outcomes from text through iteration and discussion with clinicians and Natural Language Processing experts. Then, three independent annotators annotated the Results and Conclusions sections of a randomly selected sample of 500 PubMed abstracts and 140 PubMed abstracts from the existing EBM-NLP corpus. This resulted in EvidenceOutcomes with high-quality annotations of an inter-rater agreement of 0.76. Additionally, our fine-tuned PubMedBERT model, applied to these 500 PubMed abstracts, achieved an F1-score of 0.69 at the entity level and 0.76 at the token level on the subset of 140 PubMed abstracts from the EBM-NLP corpus. EvidenceOutcomes can serve as a shared benchmark to develop and test future machine learning algorithms to extract clinically meaningful outcomes from biomedical abstracts. 

---
# Movie Facts and Fibs (MF$^2$): A Benchmark for Long Movie Understanding 

**Authors**: Emmanouil Zaranis, António Farinhas, Saul Santos, Beatriz Canaverde, Miguel Moura Ramos, Aditya K Surikuchi, André Viveiros, Baohao Liao, Elena Bueno-Benito, Nithin Sivakumaran, Pavlo Vasylenko, Shoubin Yu, Sonal Sannigrahi, Wafaa Mohammed, Ben Peters, Danae Sánchez Villegas, Elias Stengel-Eskin, Giuseppe Attanasio, Jaehong Yoon, Stella Frank, Alessandro Suglia, Chrysoula Zerva, Desmond Elliott, Mariella Dimiccoli, Mohit Bansal, Oswald Lanz, Raffaella Bernardi, Raquel Fernández, Sandro Pezzelle, Vlad Niculae, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2506.06275)  

**Abstract**: Despite recent progress in vision-language models (VLMs), holistic understanding of long-form video content remains a significant challenge, partly due to limitations in current benchmarks. Many focus on peripheral, ``needle-in-a-haystack'' details, encouraging context-insensitive retrieval over deep comprehension. Others rely on large-scale, semi-automatically generated questions (often produced by language models themselves) that are easier for models to answer but fail to reflect genuine understanding. In this paper, we introduce MF$^2$, a new benchmark for evaluating whether models can comprehend, consolidate, and recall key narrative information from full-length movies (50-170 minutes long). MF$^2$ includes over 50 full-length, open-licensed movies, each paired with manually constructed sets of claim pairs -- one true (fact) and one plausible but false (fib), totalling over 850 pairs. These claims target core narrative elements such as character motivations and emotions, causal chains, and event order, and refer to memorable moments that humans can recall without rewatching the movie. Instead of multiple-choice formats, we adopt a binary claim evaluation protocol: for each pair, models must correctly identify both the true and false claims. This reduces biases like answer ordering and enables a more precise assessment of reasoning. Our experiments demonstrate that both open-weight and closed state-of-the-art models fall well short of human performance, underscoring the relative ease of the task for humans and their superior ability to retain and reason over critical narrative information -- an ability current VLMs lack. 

---
# PersonaAgent: When Large Language Model Agents Meet Personalization at Test Time 

**Authors**: Weizhi Zhang, Xinyang Zhang, Chenwei Zhang, Liangwei Yang, Jingbo Shang, Zhepei Wei, Henry Peng Zou, Zijie Huang, Zhengyang Wang, Yifan Gao, Xiaoman Pan, Lian Xiong, Jingguo Liu, Philip S. Yu, Xian Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06254)  

**Abstract**: Large Language Model (LLM) empowered agents have recently emerged as advanced paradigms that exhibit impressive capabilities in a wide range of domains and tasks. Despite their potential, current LLM agents often adopt a one-size-fits-all approach, lacking the flexibility to respond to users' varying needs and preferences. This limitation motivates us to develop PersonaAgent, the first personalized LLM agent framework designed to address versatile personalization tasks. Specifically, PersonaAgent integrates two complementary components - a personalized memory module that includes episodic and semantic memory mechanisms; a personalized action module that enables the agent to perform tool actions tailored to the user. At the core, the persona (defined as unique system prompt for each user) functions as an intermediary: it leverages insights from personalized memory to control agent actions, while the outcomes of these actions in turn refine the memory. Based on the framework, we propose a test-time user-preference alignment strategy that simulate the latest n interactions to optimize the persona prompt, ensuring real-time user preference alignment through textual loss feedback between simulated and ground-truth responses. Experimental evaluations demonstrate that PersonaAgent significantly outperforms other baseline methods by not only personalizing the action space effectively but also scaling during test-time real-world applications. These results underscore the feasibility and potential of our approach in delivering tailored, dynamic user experiences. 

---
# Corrector Sampling in Language Models 

**Authors**: Itai Gat, Neta Shaul, Uriel Singer, Yaron Lipman  

**Link**: [PDF](https://arxiv.org/pdf/2506.06215)  

**Abstract**: Autoregressive language models accumulate errors due to their fixed, irrevocable left-to-right token generation. To address this, we propose a new sampling method called Resample-Previous-Tokens (RPT). RPT mitigates error accumulation by iteratively revisiting and potentially replacing tokens in a window of previously generated text. This method can be integrated into existing autoregressive models, preserving their next-token-prediction quality and speed. Fine-tuning a pretrained 8B parameter model with RPT for only 100B resulted in ~10% relative improvements on reasoning and coding benchmarks compared to the standard sampling. 

---
# The Lock-in Hypothesis: Stagnation by Algorithm 

**Authors**: Tianyi Alex Qiu, Zhonghao He, Tejasveer Chugh, Max Kleiman-Weiner  

**Link**: [PDF](https://arxiv.org/pdf/2506.06166)  

**Abstract**: The training and deployment of large language models (LLMs) create a feedback loop with human users: models learn human beliefs from data, reinforce these beliefs with generated content, reabsorb the reinforced beliefs, and feed them back to users again and again. This dynamic resembles an echo chamber. We hypothesize that this feedback loop entrenches the existing values and beliefs of users, leading to a loss of diversity and potentially the lock-in of false beliefs. We formalize this hypothesis and test it empirically with agent-based LLM simulations and real-world GPT usage data. Analysis reveals sudden but sustained drops in diversity after the release of new GPT iterations, consistent with the hypothesized human-AI feedback loop. Code and data available at this https URL 

---
# Masked Language Models are Good Heterogeneous Graph Generalizers 

**Authors**: Jinyu Yang, Cheng Yang, Shanyuan Cui, Zeyuan Guo, Liangwei Yang, Muhan Zhang, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06157)  

**Abstract**: Heterogeneous graph neural networks (HGNNs) excel at capturing structural and semantic information in heterogeneous graphs (HGs), while struggling to generalize across domains and tasks. Recently, some researchers have turned to integrating HGNNs with large language models (LLMs) for more generalizable heterogeneous graph learning. However, these approaches typically extract structural information via HGNNs as HG tokens, and disparities in embedding spaces between HGNNs and LLMs have been shown to bias the LLM's comprehension of HGs. Moreover, as these HG tokens are often derived from node-level tasks, the model's ability to generalize across tasks remains limited. To this end, we propose a simple yet effective Masked Language Modeling-based method, called MLM4HG. MLM4HG introduces metapath-based textual sequences instead of HG tokens to extract structural and semantic information inherent in HGs, and designs customized textual templates to unify different graph tasks into a coherent cloze-style "mask" token prediction paradigm. Specifically, MLM4HG first converts HGs from various domains to texts based on metapaths, and subsequently combines them with the unified task texts to form a HG-based corpus. Moreover, the corpus is fed into a pretrained LM for fine-tuning with a constrained target vocabulary, enabling the fine-tuned LM to generalize to unseen target HGs. Extensive cross-domain and multi-task experiments on four real-world datasets demonstrate the superior generalization performance of MLM4HG over state-of-the-art methods in both few-shot and zero-shot scenarios. Our code is available at this https URL. 

---
# CLaMR: Contextualized Late-Interaction for Multimodal Content Retrieval 

**Authors**: David Wan, Han Wang, Elias Stengel-Eskin, Jaemin Cho, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2506.06144)  

**Abstract**: Online video web content is richly multimodal: a single video blends vision, speech, ambient audio, and on-screen text. Retrieval systems typically treat these modalities as independent retrieval sources, which can lead to noisy and subpar retrieval. We explore multimodal video content retrieval, where relevance can be scored from one particular modality or jointly across multiple modalities simultaneously. Consequently, an effective retriever must dynamically choose which modality (or set of modalities) best addresses the query. We introduce CLaMR, a multimodal, late-interaction retriever that jointly indexes 4 modalities: video frames, transcribed speech, on-screen text, and metadata. CLaMR jointly encodes all modalities with a unified multimodal backbone for improved contextualization and is trained to enhance dynamic modality selection via two key innovations. First, given the lack of training data for multimodal retrieval, we introduce MultiVENT 2.0++, a large-scale synthetic training dataset built on MultiVENT 2.0 (event-centric videos in various languages paired with queries) with modality-targeted queries. Next, we propose a modality-aware loss that jointly trains according to a standard contrastive objective alongside an objective for learning correct modality usage. On the test sets of MultiVENT 2.0++ and MSRVTT, conventional aggregation strategies, such as averaging similarities for baseline retrievers, degrade performance by introducing noise from irrelevant modalities. In contrast, CLaMR consistently outperforms existing retrievers: on MultiVENT 2.0++, CLaMR improves nDCG@10 by 25.6 over the best single-modality retriever and by 35.4 over the best multi-modality retriever. We illustrate CLaMR's downstream utility on long-video QA, retrieving relevant frames and obtaining a 3.50% boost over LanguageBind on Video-MME and 1.42% over dense sampling on LongVideoBench. 

---
# Table-r1: Self-supervised and Reinforcement Learning for Program-based Table Reasoning in Small Language Models 

**Authors**: Rihui Jin, Zheyu Xin, Xing Xie, Zuoyi Li, Guilin Qi, Yongrui Chen, Xinbang Dai, Tongtong Wu, Gholamreza Haffari  

**Link**: [PDF](https://arxiv.org/pdf/2506.06137)  

**Abstract**: Table reasoning (TR) requires structured reasoning over semi-structured tabular data and remains challenging, particularly for small language models (SLMs, e.g., LLaMA-8B) due to their limited capacity compared to large LMs (LLMs, e.g., GPT-4o). To narrow this gap, we explore program-based TR (P-TR), which circumvents key limitations of text-based TR (T-TR), notably in numerical reasoning, by generating executable programs. However, applying P-TR to SLMs introduces two challenges: (i) vulnerability to heterogeneity in table layouts, and (ii) inconsistency in reasoning due to limited code generation capability. We propose Table-r1, a two-stage P-TR method designed for SLMs. Stage 1 introduces an innovative self-supervised learning task, Layout Transformation Inference, to improve tabular layout generalization from a programmatic view. Stage 2 adopts a mix-paradigm variant of Group Relative Policy Optimization, enhancing P-TR consistency while allowing dynamic fallback to T-TR when needed. Experiments on four TR benchmarks demonstrate that Table-r1 outperforms all SLM-based methods, achieving at least a 15% accuracy improvement over the base model (LLaMA-8B) across all datasets and reaching performance competitive with LLMs. 

---
# Label-Context-Dependent Internal Language Model Estimation for CTC 

**Authors**: Zijian Yang, Minh-Nghia Phan, Ralf Schlüter, Hermann Ney  

**Link**: [PDF](https://arxiv.org/pdf/2506.06096)  

**Abstract**: Although connectionist temporal classification (CTC) has the label context independence assumption, it can still implicitly learn a context-dependent internal language model (ILM) due to modern powerful encoders. In this work, we investigate the implicit context dependency modeled in the ILM of CTC. To this end, we propose novel context-dependent ILM estimation methods for CTC based on knowledge distillation (KD) with theoretical justifications. Furthermore, we introduce two regularization methods for KD. We conduct experiments on Librispeech and TED-LIUM Release 2 datasets for in-domain and cross-domain evaluation, respectively. Experimental results show that context-dependent ILMs outperform the context-independent priors in cross-domain evaluation, indicating that CTC learns a context-dependent ILM. The proposed label-level KD with smoothing method surpasses other ILM estimation approaches, with more than 13% relative improvement in word error rate compared to shallow fusion. 

---
# CO-VADA: A Confidence-Oriented Voice Augmentation Debiasing Approach for Fair Speech Emotion Recognition 

**Authors**: Yun-Shao Tsai, Yi-Cheng Lin, Huang-Cheng Chou, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.06071)  

**Abstract**: Bias in speech emotion recognition (SER) systems often stems from spurious correlations between speaker characteristics and emotional labels, leading to unfair predictions across demographic groups. Many existing debiasing methods require model-specific changes or demographic annotations, limiting their practical use. We present CO-VADA, a Confidence-Oriented Voice Augmentation Debiasing Approach that mitigates bias without modifying model architecture or relying on demographic information. CO-VADA identifies training samples that reflect bias patterns present in the training data and then applies voice conversion to alter irrelevant attributes and generate samples. These augmented samples introduce speaker variations that differ from dominant patterns in the data, guiding the model to focus more on emotion-relevant features. Our framework is compatible with various SER models and voice conversion tools, making it a scalable and practical solution for improving fairness in SER systems. 

---
# Bootstrapping World Models from Dynamics Models in Multimodal Foundation Models 

**Authors**: Yifu Qiu, Yftah Ziser, Anna Korhonen, Shay B. Cohen, Edoardo M. Ponti  

**Link**: [PDF](https://arxiv.org/pdf/2506.06006)  

**Abstract**: To what extent do vision-and-language foundation models possess a realistic world model (observation $\times$ action $\rightarrow$ observation) and a dynamics model (observation $\times$ observation $\rightarrow$ action), when actions are expressed through language? While open-source foundation models struggle with both, we find that fine-tuning them to acquire a dynamics model through supervision is significantly easier than acquiring a world model. In turn, dynamics models can be used to bootstrap world models through two main strategies: 1) weakly supervised learning from synthetic data and 2) inference time verification. Firstly, the dynamics model can annotate actions for unlabelled pairs of video frame observations to expand the training data. We further propose a new objective, where image tokens in observation pairs are weighted by their importance, as predicted by a recognition model. Secondly, the dynamics models can assign rewards to multiple samples of the world model to score them, effectively guiding search at inference time. We evaluate the world models resulting from both strategies through the task of action-centric image editing on Aurora-Bench. Our best model achieves a performance competitive with state-of-the-art image editing models, improving on them by a margin of $15\%$ on real-world subsets according to GPT4o-as-judge, and achieving the best average human evaluation across all subsets of Aurora-Bench. 

---
# Audio-Aware Large Language Models as Judges for Speaking Styles 

**Authors**: Cheng-Han Chiang, Xiaofei Wang, Chung-Ching Lin, Kevin Lin, Linjie Li, Radu Kopetz, Yao Qian, Zhendong Wang, Zhengyuan Yang, Hung-yi Lee, Lijuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05984)  

**Abstract**: Audio-aware large language models (ALLMs) can understand the textual and non-textual information in the audio input. In this paper, we explore using ALLMs as an automatic judge to assess the speaking styles of speeches. We use ALLM judges to evaluate the speeches generated by SLMs on two tasks: voice style instruction following and role-playing. The speaking style we consider includes emotion, volume, speaking pace, word emphasis, pitch control, and non-verbal elements. We use four spoken language models (SLMs) to complete the two tasks and use humans and ALLMs to judge the SLMs' responses. We compare two ALLM judges, GPT-4o-audio and Gemini-2.5-pro, with human evaluation results and show that the agreement between Gemini and human judges is comparable to the agreement between human evaluators. These promising results show that ALLMs can be used as a judge to evaluate SLMs. Our results also reveal that current SLMs, even GPT-4o-audio, still have room for improvement in controlling the speaking style and generating natural dialogues. 

---
# Proactive Assistant Dialogue Generation from Streaming Egocentric Videos 

**Authors**: Yichi Zhang, Xin Luna Dong, Zhaojiang Lin, Andrea Madotto, Anuj Kumar, Babak Damavandi, Joyce Chai, Seungwhan Moon  

**Link**: [PDF](https://arxiv.org/pdf/2506.05904)  

**Abstract**: Recent advances in conversational AI have been substantial, but developing real-time systems for perceptual task guidance remains challenging. These systems must provide interactive, proactive assistance based on streaming visual inputs, yet their development is constrained by the costly and labor-intensive process of data collection and system evaluation. To address these limitations, we present a comprehensive framework with three key contributions. First, we introduce a novel data curation pipeline that synthesizes dialogues from annotated egocentric videos, resulting in \dataset, a large-scale synthetic dialogue dataset spanning multiple domains. Second, we develop a suite of automatic evaluation metrics, validated through extensive human studies. Third, we propose an end-to-end model that processes streaming video inputs to generate contextually appropriate responses, incorporating novel techniques for handling data imbalance and long-duration videos. This work lays the foundation for developing real-time, proactive AI assistants capable of guiding users through diverse tasks. Project page: this https URL 

---
# CodeContests+: High-Quality Test Case Generation for Competitive Programming 

**Authors**: Zihan Wang, Siyao Liu, Yang Sun, Hongyan Li, Kai Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.05817)  

**Abstract**: Competitive programming, due to its high reasoning difficulty and precise correctness feedback, has become a key task for both training and evaluating the reasoning capabilities of large language models (LLMs). However, while a large amount of public problem data, such as problem statements and solutions, is available, the test cases of these problems are often difficult to obtain. Therefore, test case generation is a necessary task for building large-scale datasets, and the quality of the test cases directly determines the accuracy of the evaluation. In this paper, we introduce an LLM-based agent system that creates high-quality test cases for competitive programming problems. We apply this system to the CodeContests dataset and propose a new version with improved test cases, named CodeContests+. We evaluated the quality of test cases in CodeContestsPlus. First, we used 1.72 million submissions with pass/fail labels to examine the accuracy of these test cases in evaluation. The results indicated that CodeContests+ achieves significantly higher accuracy than CodeContests, particularly with a notably higher True Positive Rate (TPR). Subsequently, our experiments in LLM Reinforcement Learning (RL) further confirmed that improvements in test case quality yield considerable advantages for RL. 

---
# Do Large Vision-Language Models Distinguish between the Actual and Apparent Features of Illusions? 

**Authors**: Taiga Shinozaki, Tomoki Doi, Satoshi Nishida, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2506.05765)  

**Abstract**: Humans are susceptible to optical illusions, which serve as valuable tools for investigating sensory and cognitive processes. Inspired by human vision studies, research has begun exploring whether machines, such as large vision language models (LVLMs), exhibit similar susceptibilities to visual illusions. However, studies often have used non-abstract images and have not distinguished actual and apparent features, leading to ambiguous assessments of machine cognition. To address these limitations, we introduce a visual question answering (VQA) dataset, categorized into genuine and fake illusions, along with corresponding control images. Genuine illusions present discrepancies between actual and apparent features, whereas fake illusions have the same actual and apparent features even though they look illusory due to the similar geometric configuration. We evaluate the performance of LVLMs for genuine and fake illusion VQA tasks and investigate whether the models discern actual and apparent features. Our findings indicate that although LVLMs may appear to recognize illusions by correctly answering questions about both feature types, they predict the same answers for both Genuine Illusion and Fake Illusion VQA questions. This suggests that their responses might be based on prior knowledge of illusions rather than genuine visual understanding. The dataset is available at this https URL 

---
# Constrained Sampling for Language Models Should Be Easy: An MCMC Perspective 

**Authors**: Emmanuel Anaya Gonzalez, Sairam Vaidya, Kanghee Park, Ruyi Ji, Taylor Berg-Kirkpatrick, Loris D'Antoni  

**Link**: [PDF](https://arxiv.org/pdf/2506.05754)  

**Abstract**: Constrained decoding enables Language Models (LMs) to produce samples that provably satisfy hard constraints. However, existing constrained-decoding approaches often distort the underlying model distribution, a limitation that is especially problematic in applications like program fuzzing, where one wants to generate diverse and valid program inputs for testing purposes. We propose a new constrained sampling framework based on Markov Chain Monte Carlo (MCMC) that simultaneously satisfies three core desiderata: constraint satisfying (every sample satisfies the constraint), monotonically converging (the sampling process converges to the true conditional distribution), and efficient (high-quality samples emerge in few steps). Our method constructs a proposal distribution over valid outputs and applies a Metropolis-Hastings acceptance criterion based on the LM's likelihood, ensuring principled and efficient exploration of the constrained space. Empirically, our sampler outperforms existing methods on both synthetic benchmarks and real-world program fuzzing tasks. 

---
# Voice Impression Control in Zero-Shot TTS 

**Authors**: Keinichi Fujita, Shota Horiguchi, Yusuke Ijima  

**Link**: [PDF](https://arxiv.org/pdf/2506.05688)  

**Abstract**: Para-/non-linguistic information in speech is pivotal in shaping the listeners' impression. Although zero-shot text-to-speech (TTS) has achieved high speaker fidelity, modulating subtle para-/non-linguistic information to control perceived voice characteristics, i.e., impressions, remains challenging. We have therefore developed a voice impression control method in zero-shot TTS that utilizes a low-dimensional vector to represent the intensities of various voice impression pairs (e.g., dark-bright). The results of both objective and subjective evaluations have demonstrated our method's effectiveness in impression control. Furthermore, generating this vector via a large language model enables target-impression generation from a natural language description of the desired impression, thus eliminating the need for manual optimization. 

---
# Contextually Guided Transformers via Low-Rank Adaptation 

**Authors**: Andrey Zhmoginov, Jihwan Lee, Max Vladymyrov, Mark Sandler  

**Link**: [PDF](https://arxiv.org/pdf/2506.05672)  

**Abstract**: Large Language Models (LLMs) based on Transformers excel at text processing, but their reliance on prompts for specialized behavior introduces computational overhead. We propose a modification to a Transformer architecture that eliminates the need for explicit prompts by learning to encode context into the model's weights. Our Contextually Guided Transformer (CGT) model maintains a contextual summary at each sequence position, allowing it to update the weights on the fly based on the preceding context. This approach enables the model to self-specialize, effectively creating a tailored model for processing information following a given prefix. We demonstrate the effectiveness of our method on synthetic in-context learning tasks and language modeling benchmarks. Furthermore, we introduce techniques for enhancing the interpretability of the learned contextual representations, drawing connections to Variational Autoencoders and promoting smoother, more consistent context encoding. This work offers a novel direction for efficient and adaptable language modeling by integrating context directly into the model's architecture. 

---
# Low-Resource Domain Adaptation for Speech LLMs via Text-Only Fine-Tuning 

**Authors**: Yangui Fang, Jing Peng, Xu Li, Yu Xi, Chengwei Zhang, Guohui Zhong, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05671)  

**Abstract**: Recent advances in automatic speech recognition (ASR) have combined speech encoders with large language models (LLMs) through projection, forming Speech LLMs with strong performance. However, adapting them to new domains remains challenging, especially in low-resource settings where paired speech-text data is scarce. We propose a text-only fine-tuning strategy for Speech LLMs using unpaired target-domain text without requiring additional audio. To preserve speech-text alignment, we introduce a real-time evaluation mechanism during fine-tuning. This enables effective domain adaptation while maintaining source-domain performance. Experiments on LibriSpeech, SlideSpeech, and Medical datasets show that our method achieves competitive recognition performance, with minimal degradation compared to full audio-text fine-tuning. It also improves generalization to new domains without catastrophic forgetting, highlighting the potential of text-only fine-tuning for low-resource domain adaptation of ASR. 

---
# BAQ: Efficient Bit Allocation Quantization for Large Language Models 

**Authors**: Chao Zhang, Li Wang, Samson Lasaulce, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2506.05664)  

**Abstract**: Post-training model quantization is a widely adopted technique for reducing the memory and computational costs of large language models (LLMs). However, most existing methods rely on uniform or heuristic bitwidth assignments, failing to account for the nonuniform sensitivity of weights to quantization noise. In this paper, we propose a novel framework for allocating quantization bitwidths based on sensitivity metrics derived from a Hessian proxy. We make key assumptions, which allow the layer/component-wise loss function to be expressed as an explicit function of the bitwidths. This enables a neat formulation of the bit allocation problem as a convex optimization task, whose closed-form solution adapts precision across weights to minimize the layer-wise quantization loss. Inspecting the solution provides several insights (such as the equal-loss structure), which are then exploited to design the proposed \textbf{BAQ} (Bit Allocation Quantization) algorithm. The proposed algorithm achieves a good trade-off between loss minimization and complexity and allows BAQ to be integrated into standard quantization pipelines with minimal overhead. Experimental results show that BAQ consistently outperforms GPTQ, achieving up to 56$\times$ lower perplexity at the same bitwidth on large language models ranging from 125M to 30B parameters. Leveraging our analytical results derived from solving the optimal bit allocation problem, we also provide a theoretical explanation for the observed gains. All codes of this paper are available at this https URL. 

---
# Projectable Models: One-Shot Generation of Small Specialized Transformers from Large Ones 

**Authors**: Andrey Zhmoginov, Jihwan Lee, Mark Sandler  

**Link**: [PDF](https://arxiv.org/pdf/2506.05641)  

**Abstract**: Modern Foundation Models (FMs) are typically trained on corpora spanning a wide range of different data modalities, topics and downstream tasks. Utilizing these models can be very computationally expensive and is out of reach for most consumer devices. Furthermore, most of the broad FM knowledge may actually be irrelevant for a specific task at hand. Here we explore a technique for mapping parameters of a large Transformer to parameters of a smaller specialized model. By making this transformation task-specific, we aim to capture a narrower scope of the knowledge needed for performing a specific task by a smaller model. We study our method on image modeling tasks, showing that performance of generated models exceeds that of universal conditional models. 

---
# Deployability-Centric Infrastructure-as-Code Generation: An LLM-based Iterative Framework 

**Authors**: Tianyi Zhang, Shidong Pan, Zejun Zhang, Zhenchang Xing, Xiaoyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.05623)  

**Abstract**: Infrastructure-as-Code (IaC) generation holds significant promise for automating cloud infrastructure provisioning. Recent advances in Large Language Models (LLMs) present a promising opportunity to democratize IaC development by generating deployable infrastructure templates from natural language descriptions, but current evaluation focuses on syntactic correctness while ignoring deployability, the fatal measure of IaC template utility. We address this gap through two contributions: (1) IaCGen, an LLM-based deployability-centric framework that uses iterative feedback mechanism to generate IaC templates, and (2) DPIaC-Eval, a deployability-centric IaC template benchmark consists of 153 real-world scenarios that can evaluate syntax, deployment, user intent, and security. Our evaluation reveals that state-of-the-art LLMs initially performed poorly, with Claude-3.5 and Claude-3.7 achieving only 30.2% and 26.8% deployment success on the first attempt respectively. However, IaCGen transforms this performance dramatically: all evaluated models reach over 90% passItr@25, with Claude-3.5 and Claude-3.7 achieving 98% success rate. Despite these improvements, critical challenges remain in user intent alignment (25.2% accuracy) and security compliance (8.4% pass rate), highlighting areas requiring continued research. Our work provides the first comprehensive assessment of deployability-centric IaC template generation and establishes a foundation for future research. 

---
# SoK: Are Watermarks in LLMs Ready for Deployment? 

**Authors**: Kieu Dang, Phung Lai, NhatHai Phan, Yelong Shen, Ruoming Jin, Abdallah Khreishah, My Thai  

**Link**: [PDF](https://arxiv.org/pdf/2506.05594)  

**Abstract**: Large Language Models (LLMs) have transformed natural language processing, demonstrating impressive capabilities across diverse tasks. However, deploying these models introduces critical risks related to intellectual property violations and potential misuse, particularly as adversaries can imitate these models to steal services or generate misleading outputs. We specifically focus on model stealing attacks, as they are highly relevant to proprietary LLMs and pose a serious threat to their security, revenue, and ethical deployment. While various watermarking techniques have emerged to mitigate these risks, it remains unclear how far the community and industry have progressed in developing and deploying watermarks in LLMs.
To bridge this gap, we aim to develop a comprehensive systematization for watermarks in LLMs by 1) presenting a detailed taxonomy for watermarks in LLMs, 2) proposing a novel intellectual property classifier to explore the effectiveness and impacts of watermarks on LLMs under both attack and attack-free environments, 3) analyzing the limitations of existing watermarks in LLMs, and 4) discussing practical challenges and potential future directions for watermarks in LLMs. Through extensive experiments, we show that despite promising research outcomes and significant attention from leading companies and community to deploy watermarks, these techniques have yet to reach their full potential in real-world applications due to their unfavorable impacts on model utility of LLMs and downstream tasks. Our findings provide an insightful understanding of watermarks in LLMs, highlighting the need for practical watermarks solutions tailored to LLM deployment. 

---
# MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark 

**Authors**: Junjie Xing, Yeye He, Mengyu Zhou, Haoyu Dong, Shi Han, Lingjiao Chen, Dongmei Zhang, Surajit Chaudhuri, H. V. Jagadish  

**Link**: [PDF](https://arxiv.org/pdf/2506.05587)  

**Abstract**: Tables and table-based use cases play a crucial role in many important real-world applications, such as spreadsheets, databases, and computational notebooks, which traditionally require expert-level users like data engineers, data analysts, and database administrators to operate. Although LLMs have shown remarkable progress in working with tables (e.g., in spreadsheet and database copilot scenarios), comprehensive benchmarking of such capabilities remains limited. In contrast to an extensive and growing list of NLP benchmarks, evaluations of table-related tasks are scarce, and narrowly focus on tasks like NL-to-SQL and Table-QA, overlooking the broader spectrum of real-world tasks that professional users face. This gap limits our understanding and model progress in this important area.
In this work, we introduce MMTU, a large-scale benchmark with over 30K questions across 25 real-world table tasks, designed to comprehensively evaluate models ability to understand, reason, and manipulate real tables at the expert-level. These tasks are drawn from decades' worth of computer science research on tabular data, with a focus on complex table tasks faced by professional users. We show that MMTU require a combination of skills -- including table understanding, reasoning, and coding -- that remain challenging for today's frontier models, where even frontier reasoning models like OpenAI o4-mini and DeepSeek R1 score only around 60%, suggesting significant room for improvement. We highlight key findings in our evaluation using MMTU and hope that this benchmark drives further advances in understanding and developing foundation models for structured data processing and analysis. Our code and data are available at this https URL and this https URL. 

---
# When Models Know More Than They Can Explain: Quantifying Knowledge Transfer in Human-AI Collaboration 

**Authors**: Quan Shi, Carlos E. Jimenez, Shunyu Yao, Nick Haber, Diyi Yang, Karthik Narasimhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.05579)  

**Abstract**: Recent advancements in AI reasoning have driven substantial improvements across diverse tasks. A critical open question is whether these improvements also yields better knowledge transfer: the ability of models to communicate reasoning in ways humans can understand, apply, and learn from. To investigate this, we introduce Knowledge Integration and Transfer Evaluation (KITE), a conceptual and experimental framework for Human-AI knowledge transfer capabilities and conduct the first large-scale human study (N=118) explicitly designed to measure it. In our two-phase setup, humans first ideate with an AI on problem-solving strategies, then independently implement solutions, isolating model explanations' influence on human understanding. Our findings reveal that although model benchmark performance correlates with collaborative outcomes, this relationship is notably inconsistent, featuring significant outliers, indicating that knowledge transfer requires dedicated optimization. Our analysis identifies behavioral and strategic factors mediating successful knowledge transfer. We release our code, dataset, and evaluation framework to support future work on communicatively aligned models. 

---
# MORSE-500: A Programmatically Controllable Video Benchmark to Stress-Test Multimodal Reasoning 

**Authors**: Zikui Cai, Andrew Wang, Anirudh Satheesh, Ankit Nakhawa, Hyunwoo Jae, Keenan Powell, Minghui Liu, Neel Jay, Sungbin Oh, Xiyao Wang, Yongyuan Liang, Tom Goldstein, Furong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05523)  

**Abstract**: Despite rapid advances in vision-language models (VLMs), current benchmarks for multimodal reasoning fall short in three key dimensions. First, they overwhelmingly rely on static images, failing to capture the temporal complexity of real-world environments. Second, they narrowly focus on mathematical problem-solving, neglecting the broader spectrum of reasoning skills -- including abstract, physical, planning, spatial, and temporal capabilities -- required for robust multimodal intelligence. Third, many benchmarks quickly saturate, offering limited headroom for diagnosing failure modes or measuring continued progress. We introduce MORSE-500 (Multimodal Reasoning Stress-test Environment), a video benchmark composed of 500 fully scripted clips with embedded questions spanning six complementary reasoning categories. Each instance is programmatically generated using deterministic Python scripts (via Manim, Matplotlib, MoviePy), generative video models, and curated real footage. This script-driven design allows fine-grained control over visual complexity, distractor density, and temporal dynamics -- enabling difficulty to be scaled systematically as models improve. Unlike static benchmarks that become obsolete once saturated, MORSE-500 is built to evolve: its controllable generation pipeline supports the creation of arbitrarily challenging new instances, making it ideally suited for stress-testing next-generation models. Initial experiments with state-of-the-art systems -- including various Gemini 2.5 Pro and OpenAI o3 which represent the strongest available at the time, alongside strong open-source models -- reveal substantial performance gaps across all categories, with particularly large deficits in abstract and planning tasks. We release the full dataset, generation scripts, and evaluation harness to support transparent, reproducible, and forward-looking multimodal reasoning research. 

---
# Interpretation Meets Safety: A Survey on Interpretation Methods and Tools for Improving LLM Safety 

**Authors**: Seongmin Lee, Aeree Cho, Grace C. Kim, ShengYun Peng, Mansi Phute, Duen Horng Chau  

**Link**: [PDF](https://arxiv.org/pdf/2506.05451)  

**Abstract**: As large language models (LLMs) see wider real-world use, understanding and mitigating their unsafe behaviors is critical. Interpretation techniques can reveal causes of unsafe outputs and guide safety, but such connections with safety are often overlooked in prior surveys. We present the first survey that bridges this gap, introducing a unified framework that connects safety-focused interpretation methods, the safety enhancements they inform, and the tools that operationalize them. Our novel taxonomy, organized by LLM workflow stages, summarizes nearly 70 works at their intersections. We conclude with open challenges and future directions. This timely survey helps researchers and practitioners navigate key advancements for safer, more interpretable LLMs. 

---
# BYO-Eval: Build Your Own Dataset for Fine-Grained Visual Assessment of Multimodal Language Models 

**Authors**: Ludovic Arnould, Salim Khazem, Hugues Ali Mehenni  

**Link**: [PDF](https://arxiv.org/pdf/2506.05440)  

**Abstract**: Visual Language Models (VLMs) are now sufficiently advanced to support a broad range of applications, including answering complex visual questions, and are increasingly expected to interact with images in varied ways. To evaluate them, current benchmarks often focus on specific domains (e.g., reading charts), constructing datasets of annotated real images paired with pre-defined Multiple Choice Questions (MCQs) to report aggregate accuracy scores. However, such benchmarks entail high annotation costs, risk information leakage, and do not clarify whether failures stem from limitations in visual perception, reasoning, or general knowledge. We propose a new evaluation methodology, inspired by ophthalmologic diagnostics, leveraging procedural generation of synthetic images to obtain control over visual attributes and precisely reveal perception failures in VLMs. Specifically, we build collections of images with gradually more challenging variations in the content of interest (e.g., number of objects in a counting task) while holding other visual parameters constant. This diagnostic allows systematic stress testing and fine-grained failure analysis, shifting the focus from coarse benchmarking toward targeted and interpretable assessment of VLM capabilities. Our code is available at this https URL. 

---
# LLMs Can Compensate for Deficiencies in Visual Representations 

**Authors**: Sho Takishita, Jay Gala, Abdelrahman Mohamed, Kentaro Inui, Yova Kementchedjhieva  

**Link**: [PDF](https://arxiv.org/pdf/2506.05439)  

**Abstract**: Many vision-language models (VLMs) that prove very effective at a range of multimodal task, build on CLIP-based vision encoders, which are known to have various limitations. We investigate the hypothesis that the strong language backbone in VLMs compensates for possibly weak visual features by contextualizing or enriching them. Using three CLIP-based VLMs, we perform controlled self-attention ablations on a carefully designed probing task. Our findings show that despite known limitations, CLIP visual representations offer ready-to-read semantic information to the language decoder. However, in scenarios of reduced contextualization in the visual representations, the language decoder can largely compensate for the deficiency and recover performance. This suggests a dynamic division of labor in VLMs and motivates future architectures that offload more visual processing to the language decoder. 

---
# Coordinated Robustness Evaluation Framework for Vision-Language Models 

**Authors**: Ashwin Ramesh Babu, Sajad Mousavi, Vineet Gundecha, Sahand Ghorbanpour, Avisek Naug, Antonio Guillen, Ricardo Luna Gutierrez, Soumyendu Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2506.05429)  

**Abstract**: Vision-language models, which integrate computer vision and natural language processing capabilities, have demonstrated significant advancements in tasks such as image captioning and visual question and answering. However, similar to traditional models, they are susceptible to small perturbations, posing a challenge to their robustness, particularly in deployment scenarios. Evaluating the robustness of these models requires perturbations in both the vision and language modalities to learn their inter-modal dependencies. In this work, we train a generic surrogate model that can take both image and text as input and generate joint representation which is further used to generate adversarial perturbations for both the text and image modalities. This coordinated attack strategy is evaluated on the visual question and answering and visual reasoning datasets using various state-of-the-art vision-language models. Our results indicate that the proposed strategy outperforms other multi-modal attacks and single-modality attacks from the recent literature. Our results demonstrate their effectiveness in compromising the robustness of several state-of-the-art pre-trained multi-modal models such as instruct-BLIP, ViLT and others. 

---
# Can Vision Language Models Infer Human Gaze Direction? A Controlled Study 

**Authors**: Zory Zhang, Pinyuan Feng, Bingyang Wang, Tianwei Zhao, Suyang Yu, Qingying Gao, Hokin Deng, Ziqiao Ma, Yijiang Li, Dezhi Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.05412)  

**Abstract**: Gaze-referential inference--the ability to infer what others are looking at--is a critical component of a theory of mind that underpins natural human-AI interaction. In a controlled study, we evaluated this skill across 111 Vision Language Models (VLMs) using photos taken with manipulated difficulty and variability, comparing performance with that of human participants (N = 65), and analyzed behaviors using mixed-effects models. We found that 94 of the 111 VLMs failed to do better than random guessing, while humans achieved near-ceiling accuracy. VLMs even respond with each choice almost equally frequently. Are they randomly guessing? Although most VLMs struggle, when we zoom in on five of the top-tier VLMs with above-chance performance, we find that their performance declined with increasing task difficulty but varied only slightly across different prompts and scene objects. These behavioral features cannot be explained by considering them as random guessers. Instead, they likely use a combination of heuristics and guessing such that their performance is subject to the task difficulty but robust to perceptual variations. This suggests that VLMs, lacking gaze inference capability, have yet to become technologies that can naturally interact with humans, but the potential remains. 

---
# Attention-based transformer models for image captioning across languages: An in-depth survey and evaluation 

**Authors**: Israa A. Albadarneh, Bassam H. Hammo, Omar S. Al-Kadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05399)  

**Abstract**: Image captioning involves generating textual descriptions from input images, bridging the gap between computer vision and natural language processing. Recent advancements in transformer-based models have significantly improved caption generation by leveraging attention mechanisms for better scene understanding. While various surveys have explored deep learning-based approaches for image captioning, few have comprehensively analyzed attention-based transformer models across multiple languages. This survey reviews attention-based image captioning models, categorizing them into transformer-based, deep learning-based, and hybrid approaches. It explores benchmark datasets, discusses evaluation metrics such as BLEU, METEOR, CIDEr, and ROUGE, and highlights challenges in multilingual captioning. Additionally, this paper identifies key limitations in current models, including semantic inconsistencies, data scarcity in non-English languages, and limitations in reasoning ability. Finally, we outline future research directions, such as multimodal learning, real-time applications in AI-powered assistants, healthcare, and forensic analysis. This survey serves as a comprehensive reference for researchers aiming to advance the field of attention-based image captioning. 

---
