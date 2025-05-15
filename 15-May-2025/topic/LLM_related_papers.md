# Qwen3 Technical Report 

**Authors**: An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, Zihan Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09388)  

**Abstract**: In this work, we present Qwen3, the latest version of the Qwen model family. Qwen3 comprises a series of large language models (LLMs) designed to advance performance, efficiency, and multilingual capabilities. The Qwen3 series includes models of both dense and Mixture-of-Expert (MoE) architectures, with parameter scales ranging from 0.6 to 235 billion. A key innovation in Qwen3 is the integration of thinking mode (for complex, multi-step reasoning) and non-thinking mode (for rapid, context-driven responses) into a unified framework. This eliminates the need to switch between different models--such as chat-optimized models (e.g., GPT-4o) and dedicated reasoning models (e.g., QwQ-32B)--and enables dynamic mode switching based on user queries or chat templates. Meanwhile, Qwen3 introduces a thinking budget mechanism, allowing users to allocate computational resources adaptively during inference, thereby balancing latency and performance based on task complexity. Moreover, by leveraging the knowledge from the flagship models, we significantly reduce the computational resources required to build smaller-scale models, while ensuring their highly competitive performance. Empirical evaluations demonstrate that Qwen3 achieves state-of-the-art results across diverse benchmarks, including tasks in code generation, mathematical reasoning, agent tasks, etc., competitive against larger MoE models and proprietary models. Compared to its predecessor Qwen2.5, Qwen3 expands multilingual support from 29 to 119 languages and dialects, enhancing global accessibility through improved cross-lingual understanding and generation capabilities. To facilitate reproducibility and community-driven research and development, all Qwen3 models are publicly accessible under Apache 2.0. 

---
# WorldView-Bench: A Benchmark for Evaluating Global Cultural Perspectives in Large Language Models 

**Authors**: Abdullah Mushtaq, Imran Taj, Rafay Naeem, Ibrahim Ghaznavi, Junaid Qadir  

**Link**: [PDF](https://arxiv.org/pdf/2505.09595)  

**Abstract**: Large Language Models (LLMs) are predominantly trained and aligned in ways that reinforce Western-centric epistemologies and socio-cultural norms, leading to cultural homogenization and limiting their ability to reflect global civilizational plurality. Existing benchmarking frameworks fail to adequately capture this bias, as they rely on rigid, closed-form assessments that overlook the complexity of cultural inclusivity. To address this, we introduce WorldView-Bench, a benchmark designed to evaluate Global Cultural Inclusivity (GCI) in LLMs by analyzing their ability to accommodate diverse worldviews. Our approach is grounded in the Multiplex Worldview proposed by Senturk et al., which distinguishes between Uniplex models, reinforcing cultural homogenization, and Multiplex models, which integrate diverse perspectives. WorldView-Bench measures Cultural Polarization, the exclusion of alternative perspectives, through free-form generative evaluation rather than conventional categorical benchmarks. We implement applied multiplexity through two intervention strategies: (1) Contextually-Implemented Multiplex LLMs, where system prompts embed multiplexity principles, and (2) Multi-Agent System (MAS)-Implemented Multiplex LLMs, where multiple LLM agents representing distinct cultural perspectives collaboratively generate responses. Our results demonstrate a significant increase in Perspectives Distribution Score (PDS) entropy from 13% at baseline to 94% with MAS-Implemented Multiplex LLMs, alongside a shift toward positive sentiment (67.7%) and enhanced cultural balance. These findings highlight the potential of multiplex-aware AI evaluation in mitigating cultural bias in LLMs, paving the way for more inclusive and ethically aligned AI systems. 

---
# Scent of Knowledge: Optimizing Search-Enhanced Reasoning with Information Foraging 

**Authors**: Hongjin Qian, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09316)  

**Abstract**: Augmenting large language models (LLMs) with external retrieval has become a standard method to address their inherent knowledge cutoff limitations. However, traditional retrieval-augmented generation methods employ static, pre-inference retrieval strategies, making them inadequate for complex tasks involving ambiguous, multi-step, or evolving information needs. Recent advances in test-time scaling techniques have demonstrated significant potential in enabling LLMs to dynamically interact with external tools, motivating the shift toward adaptive inference-time retrieval. Inspired by Information Foraging Theory (IFT), we propose InForage, a reinforcement learning framework that formalizes retrieval-augmented reasoning as a dynamic information-seeking process. Unlike existing approaches, InForage explicitly rewards intermediate retrieval quality, encouraging LLMs to iteratively gather and integrate information through adaptive search behaviors. To facilitate training, we construct a human-guided dataset capturing iterative search and reasoning trajectories for complex, real-world web tasks. Extensive evaluations across general question answering, multi-hop reasoning tasks, and a newly developed real-time web QA dataset demonstrate InForage's superior performance over baseline methods. These results highlight InForage's effectiveness in building robust, adaptive, and efficient reasoning agents. 

---
# S-DAT: A Multilingual, GenAI-Driven Framework for Automated Divergent Thinking Assessment 

**Authors**: Jennifer Haase, Paul H. P. Hanel, Sebastian Pokutta  

**Link**: [PDF](https://arxiv.org/pdf/2505.09068)  

**Abstract**: This paper introduces S-DAT (Synthetic-Divergent Association Task), a scalable, multilingual framework for automated assessment of divergent thinking (DT) -a core component of human creativity. Traditional creativity assessments are often labor-intensive, language-specific, and reliant on subjective human ratings, limiting their scalability and cross-cultural applicability. In contrast, S-DAT leverages large language models and advanced multilingual embeddings to compute semantic distance -- a language-agnostic proxy for DT. We evaluate S-DAT across eleven diverse languages, including English, Spanish, German, Russian, Hindi, and Japanese (Kanji, Hiragana, Katakana), demonstrating robust and consistent scoring across linguistic contexts. Unlike prior DAT approaches, the S-DAT shows convergent validity with other DT measures and correct discriminant validity with convergent thinking. This cross-linguistic flexibility allows for more inclusive, global-scale creativity research, addressing key limitations of earlier approaches. S-DAT provides a powerful tool for fairer, more comprehensive evaluation of cognitive flexibility in diverse populations and can be freely assessed online: this https URL. 

---
# CEC-Zero: Chinese Error Correction Solution Based on LLM 

**Authors**: Sophie Zhang, Zhiming Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.09082)  

**Abstract**: Recent advancements in large language models (LLMs) demonstrate exceptional Chinese text processing capabilities, particularly in Chinese Spelling Correction (CSC). While LLMs outperform traditional BERT-based models in accuracy and robustness, challenges persist in reliability and generalization. This paper proposes CEC-Zero, a novel reinforcement learning (RL) framework enabling LLMs to self-correct through autonomous error strategy learning without external supervision. By integrating RL with LLMs' generative power, the method eliminates dependency on annotated data or auxiliary models. Experiments reveal RL-enhanced LLMs achieve industry-viable accuracy and superior cross-domain generalization, offering a scalable solution for reliability optimization in Chinese NLP applications. This breakthrough facilitates LLM deployment in practical Chinese text correction scenarios while establishing a new paradigm for self-improving language models. 

---
# Atomic Consistency Preference Optimization for Long-Form Question Answering 

**Authors**: Jingfeng Chen, Raghuveer Thirukovalluru, Junlin Wang, Kaiwei Luo, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2505.09039)  

**Abstract**: Large Language Models (LLMs) frequently produce factoid hallucinations - plausible yet incorrect answers. A common mitigation strategy is model alignment, which improves factual accuracy by training on curated factual and non-factual pairs. However, this approach often relies on a stronger model (e.g., GPT-4) or an external knowledge base to assess factual correctness, which may not always be accessible. To address this, we propose Atomic Consistency Preference Optimization (ACPO), a self-supervised preference-tuning method that enhances factual accuracy without external supervision. ACPO leverages atomic consistency signals, i.e., the agreement of individual facts across multiple stochastic responses, to identify high- and low-quality data pairs for model alignment. By eliminating the need for costly GPT calls, ACPO provides a scalable and efficient approach to improving factoid question-answering. Despite being self-supervised, empirical results demonstrate that ACPO outperforms FactAlign, a strong supervised alignment baseline, by 1.95 points on the LongFact and BioGen datasets, highlighting its effectiveness in enhancing factual reliability without relying on external models or knowledge bases. 

---
# A suite of LMs comprehend puzzle statements as well as humans 

**Authors**: Adele E Goldberg, Supantho Rakshit, Jennifer Hu, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2505.08996)  

**Abstract**: Recent claims suggest that large language models (LMs) underperform humans in comprehending minimally complex English statements (Dentella et al., 2024). Here, we revisit those findings and argue that human performance was overestimated, while LLM abilities were underestimated. Using the same stimuli, we report a preregistered study comparing human responses in two conditions: one allowed rereading (replicating the original study), and one that restricted rereading (a more naturalistic comprehension test). Human accuracy dropped significantly when rereading was restricted (73%), falling below that of Falcon-180B-Chat (76%) and GPT-4 (81%). The newer GPT-o1 model achieves perfect accuracy. Results further show that both humans and models are disproportionately challenged by queries involving potentially reciprocal actions (e.g., kissing), suggesting shared pragmatic sensitivities rather than model-specific deficits. Additional analyses using Llama-2-70B log probabilities, a recoding of open-ended model responses, and grammaticality ratings of other sentences reveal systematic underestimation of model performance. We find that GPT-4o can align with either naive or expert grammaticality judgments, depending on prompt framing. These findings underscore the need for more careful experimental design and coding practices in LLM evaluation, and they challenge the assumption that current models are inherently weaker than humans at language comprehension. 

---
# Customizing a Large Language Model for VHDL Design of High-Performance Microprocessors 

**Authors**: Nicolas Dupuis, Ravi Nair, Shyam Ramji, Sean McClintock, Nishant Chauhan, Priyanka Nagpal, Bart Blaner, Ken Valk, Leon Stok, Ruchir Puri  

**Link**: [PDF](https://arxiv.org/pdf/2505.09610)  

**Abstract**: The use of Large Language Models (LLMs) in hardware design has taken off in recent years, principally through its incorporation in tools that increase chip designer productivity. There has been considerable discussion about the use of LLMs in RTL specifications of chip designs, for which the two most popular languages are Verilog and VHDL. LLMs and their use in Verilog design has received significant attention due to the higher popularity of the language, but little attention so far has been given to VHDL despite its continued popularity in the industry. There has also been little discussion about the unique needs of organizations that engage in high-performance processor design, and techniques to deploy AI solutions in these settings. In this paper, we describe our journey in developing a Large Language Model (LLM) specifically for the purpose of explaining VHDL code, a task that has particular importance in an organization with decades of experience and assets in high-performance processor design. We show how we developed test sets specific to our needs and used them for evaluating models as we performed extended pretraining (EPT) of a base LLM. Expert evaluation of the code explanations produced by the EPT model increased to 69% compared to a base model rating of 43%. We further show how we developed an LLM-as-a-judge to gauge models similar to expert evaluators. This led us to deriving and evaluating a host of new models, including an instruction-tuned version of the EPT model with an expected expert evaluator rating of 71%. Our experiments also indicate that with the potential use of newer base models, this rating can be pushed to 85% and beyond. We conclude with a discussion on further improving the quality of hardware design LLMs using exciting new developments in the Generative AI world. 

---
# Ornithologist: Towards Trustworthy "Reasoning" about Central Bank Communications 

**Authors**: Dominic Zaun Eu Jones  

**Link**: [PDF](https://arxiv.org/pdf/2505.09083)  

**Abstract**: I develop Ornithologist, a weakly-supervised textual classification system and measure the hawkishness and dovishness of central bank text. Ornithologist uses ``taxonomy-guided reasoning'', guiding a large language model with human-authored decision trees. This increases the transparency and explainability of the system and makes it accessible to non-experts. It also reduces hallucination risk. Since it requires less supervision than traditional classification systems, it can more easily be applied to other problems or sources of text (e.g. news) without much modification. Ornithologist measurements of hawkishness and dovishness of RBA communication carry information about the future of the cash rate path and of market expectations. 

---
# CXMArena: Unified Dataset to benchmark performance in realistic CXM Scenarios 

**Authors**: Raghav Garg, Kapil Sharma, Karan Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.09436)  

**Abstract**: Large Language Models (LLMs) hold immense potential for revolutionizing Customer Experience Management (CXM), particularly in contact center operations. However, evaluating their practical utility in complex operational environments is hindered by data scarcity (due to privacy concerns) and the limitations of current benchmarks. Existing benchmarks often lack realism, failing to incorporate deep knowledge base (KB) integration, real-world noise, or critical operational tasks beyond conversational fluency. To bridge this gap, we introduce CXMArena, a novel, large-scale synthetic benchmark dataset specifically designed for evaluating AI in operational CXM contexts. Given the diversity in possible contact center features, we have developed a scalable LLM-powered pipeline that simulates the brand's CXM entities that form the foundation of our datasets-such as knowledge articles including product specifications, issue taxonomies, and contact center conversations. The entities closely represent real-world distribution because of controlled noise injection (informed by domain experts) and rigorous automated validation. Building on this, we release CXMArena, which provides dedicated benchmarks targeting five important operational tasks: Knowledge Base Refinement, Intent Prediction, Agent Quality Adherence, Article Search, and Multi-turn RAG with Integrated Tools. Our baseline experiments underscore the benchmark's difficulty: even state of the art embedding and generation models achieve only 68% accuracy on article search, while standard embedding methods yield a low F1 score of 0.3 for knowledge base refinement, highlighting significant challenges for current models necessitating complex pipelines and solutions over conventional techniques. 

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
# Focus, Merge, Rank: Improved Question Answering Based on Semi-structured Knowledge Bases 

**Authors**: Derian Boer, Stephen Roth, Stefan Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2505.09246)  

**Abstract**: In many real-world settings, machine learning models and interactive systems have access to both structured knowledge, e.g., knowledge graphs or tables, and unstructured content, e.g., natural language documents. However, most rely on either. Semi-Structured Knowledge Bases (SKBs) bridge this gap by linking unstructured content to nodes within structured data, thereby enabling new strategies for knowledge access and use. In this work, we present FocusedRetriever, a modular SKB-based framework for multi-hop question answering. It integrates components (VSS-based entity search, LLM-based generation of Cypher queries and pairwise re-ranking) in a way that enables it to outperform state-of-the-art methods across all three STaRK benchmark test sets, covering diverse domains and multiple performance metrics. The average first-hit rate exceeds that of the second-best method by 25.7%. FocusedRetriever leverages (1) the capacity of Large Language Models (LLMs) to extract relational facts and entity attributes from unstructured text, (2) node set joins to filter answer candidates based on these extracted triplets and constraints, (3) vector similarity search to retrieve and rank relevant unstructured content, and (4) the contextual capabilities of LLMs to finally rank the top-k answers. For generality, we only incorporate base LLMs in FocusedRetriever in our evaluation. However, our analysis of intermediate results highlights several opportunities for further upgrades including finetuning. The source code is publicly available at this https URL . 

---
# An Extra RMSNorm is All You Need for Fine Tuning to 1.58 Bits 

**Authors**: Cody Steinmetz, Gavin Childress, Aaron Herbst, Gavin Jones, Jasdeep Singh, Eli Vang, Keagan Weinstock  

**Link**: [PDF](https://arxiv.org/pdf/2505.08823)  

**Abstract**: Large language models (LLMs) have transformed natural-language processing, yet their scale makes real-world deployment costly. Post-training quantization reduces memory and computation but often degrades accuracy, while quantization-aware training can recover performance at the cost of extra training. Pushing quantization to the ternary (2-bit) regime yields even larger savings but is notoriously unstable. Building on recent work showing that a bias-free, RMS-normalized Transformer with straight-through estimation can reach 1.58-bit precision, we demonstrate that simply inserting RMS normalization before every linear projection and applying a gradual, layer-wise quantization schedule stably fine-tunes full-precision checkpoints into ternary LLMs. Our approach matches or surpasses more elaborate knowledge-distillation pipelines on standard language-modeling benchmarks without adding model complexity. These results indicate that careful normalization alone can close much of the accuracy gap between ternary and full-precision LLMs, making ultra-low-bit inference practical. 

---
# Language Agents Mirror Human Causal Reasoning Biases. How Can We Help Them Think Like Scientists? 

**Authors**: Anthony GX-Chen, Dongyan Lin, Mandana Samiei, Doina Precup, Blake A. Richards, Rob Fergus, Kenneth Marino  

**Link**: [PDF](https://arxiv.org/pdf/2505.09614)  

**Abstract**: Language model (LM) agents are increasingly used as autonomous decision-makers who need to actively gather information to guide their decisions. A crucial cognitive skill for such agents is the efficient exploration and understanding of the causal structure of the world -- key to robust, scientifically grounded reasoning. Yet, it remains unclear whether LMs possess this capability or exhibit systematic biases leading to erroneous conclusions. In this work, we examine LMs' ability to explore and infer causal relationships, using the well-established "Blicket Test" paradigm from developmental psychology. We find that LMs reliably infer the common, intuitive disjunctive causal relationships but systematically struggle with the unusual, yet equally (or sometimes even more) evidenced conjunctive ones. This "disjunctive bias" persists across model families, sizes, and prompting strategies, and performance further declines as task complexity increases. Interestingly, an analogous bias appears in human adults, suggesting that LMs may have inherited deep-seated reasoning heuristics from their training data. To this end, we quantify similarities between LMs and humans, finding that LMs exhibit adult-like inference profiles (but not children-like). Finally, we propose a test-time sampling method which explicitly samples and eliminates hypotheses about causal relationships from the LM. This scalable approach significantly reduces the disjunctive bias and moves LMs closer to the goal of scientific, causally rigorous reasoning. 

---
# Llama See, Llama Do: A Mechanistic Perspective on Contextual Entrainment and Distraction in LLMs 

**Authors**: Jingcheng Niu, Xingdi Yuan, Tong Wang, Hamidreza Saghir, Amir H. Abdi  

**Link**: [PDF](https://arxiv.org/pdf/2505.09338)  

**Abstract**: We observe a novel phenomenon, contextual entrainment, across a wide range of language models (LMs) and prompt settings, providing a new mechanistic perspective on how LMs become distracted by ``irrelevant'' contextual information in the input prompt. Specifically, LMs assign significantly higher logits (or probabilities) to any tokens that have previously appeared in the context prompt, even for random tokens. This suggests that contextual entrainment is a mechanistic phenomenon, occurring independently of the relevance or semantic relation of the tokens to the question or the rest of the sentence. We find statistically significant evidence that the magnitude of contextual entrainment is influenced by semantic factors. Counterfactual prompts have a greater effect compared to factual ones, suggesting that while contextual entrainment is a mechanistic phenomenon, it is modulated by semantic factors.
We hypothesise that there is a circuit of attention heads -- the entrainment heads -- that corresponds to the contextual entrainment phenomenon. Using a novel entrainment head discovery method based on differentiable masking, we identify these heads across various settings. When we ``turn off'' these heads, i.e., set their outputs to zero, the effect of contextual entrainment is significantly attenuated, causing the model to generate output that capitulates to what it would produce if no distracting context were provided. Our discovery of contextual entrainment, along with our investigation into LM distraction via the entrainment heads, marks a key step towards the mechanistic analysis and mitigation of the distraction problem. 

---
# The Influence of Human-inspired Agentic Sophistication in LLM-driven Strategic Reasoners 

**Authors**: Vince Trencsenyi, Agnieszka Mensfelt, Kostas Stathis  

**Link**: [PDF](https://arxiv.org/pdf/2505.09396)  

**Abstract**: The rapid rise of large language models (LLMs) has shifted artificial intelligence (AI) research toward agentic systems, motivating the use of weaker and more flexible notions of agency. However, this shift raises key questions about the extent to which LLM-based agents replicate human strategic reasoning, particularly in game-theoretic settings. In this context, we examine the role of agentic sophistication in shaping artificial reasoners' performance by evaluating three agent designs: a simple game-theoretic model, an unstructured LLM-as-agent model, and an LLM integrated into a traditional agentic framework. Using guessing games as a testbed, we benchmarked these agents against human participants across general reasoning patterns and individual role-based objectives. Furthermore, we introduced obfuscated game scenarios to assess agents' ability to generalise beyond training distributions. Our analysis, covering over 2000 reasoning samples across 25 agent configurations, shows that human-inspired cognitive structures can enhance LLM agents' alignment with human strategic behaviour. Still, the relationship between agentic design complexity and human-likeness is non-linear, highlighting a critical dependence on underlying LLM capabilities and suggesting limits to simple architectural augmentation. 

---
# Reproducibility Study of "Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents" 

**Authors**: Pedro M. P. Curvo, Mara Dragomir, Salvador Torpes, Mohammadmahdi Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2505.09289)  

**Abstract**: This study evaluates and extends the findings made by Piatti et al., who introduced GovSim, a simulation framework designed to assess the cooperative decision-making capabilities of large language models (LLMs) in resource-sharing scenarios. By replicating key experiments, we validate claims regarding the performance of large models, such as GPT-4-turbo, compared to smaller models. The impact of the universalization principle is also examined, with results showing that large models can achieve sustainable cooperation, with or without the principle, while smaller models fail without it. In addition, we provide multiple extensions to explore the applicability of the framework to new settings. We evaluate additional models, such as DeepSeek-V3 and GPT-4o-mini, to test whether cooperative behavior generalizes across different architectures and model sizes. Furthermore, we introduce new settings: we create a heterogeneous multi-agent environment, study a scenario using Japanese instructions, and explore an "inverse environment" where agents must cooperate to mitigate harmful resource distributions. Our results confirm that the benchmark can be applied to new models, scenarios, and languages, offering valuable insights into the adaptability of LLMs in complex cooperative tasks. Moreover, the experiment involving heterogeneous multi-agent systems demonstrates that high-performing models can influence lower-performing ones to adopt similar behaviors. This finding has significant implications for other agent-based applications, potentially enabling more efficient use of computational resources and contributing to the development of more effective cooperative AI systems. 

---
# Deploying Foundation Model-Enabled Air and Ground Robots in the Field: Challenges and Opportunities 

**Authors**: Zachary Ravichandran, Fernando Cladera, Jason Hughes, Varun Murali, M. Ani Hsieh, George J. Pappas, Camillo J. Taylor, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.09477)  

**Abstract**: The integration of foundation models (FMs) into robotics has enabled robots to understand natural language and reason about the semantics in their environments. However, existing FM-enabled robots primary operate in closed-world settings, where the robot is given a full prior map or has a full view of its workspace. This paper addresses the deployment of FM-enabled robots in the field, where missions often require a robot to operate in large-scale and unstructured environments. To effectively accomplish these missions, robots must actively explore their environments, navigate obstacle-cluttered terrain, handle unexpected sensor inputs, and operate with compute constraints. We discuss recent deployments of SPINE, our LLM-enabled autonomy framework, in field robotic settings. To the best of our knowledge, we present the first demonstration of large-scale LLM-enabled robot planning in unstructured environments with several kilometers of missions. SPINE is agnostic to a particular LLM, which allows us to distill small language models capable of running onboard size, weight and power (SWaP) limited platforms. Via preliminary model distillation work, we then present the first language-driven UAV planner using on-device language models. We conclude our paper by proposing several promising directions for future research. 

---
# Evaluating GPT- and Reasoning-based Large Language Models on Physics Olympiad Problems: Surpassing Human Performance and Implications for Educational Assessment 

**Authors**: Paul Tschisgale, Holger Maus, Fabian Kieser, Ben Kroehs, Stefan Petersen, Peter Wulff  

**Link**: [PDF](https://arxiv.org/pdf/2505.09438)  

**Abstract**: Large language models (LLMs) are now widely accessible, reaching learners at all educational levels. This development has raised concerns that their use may circumvent essential learning processes and compromise the integrity of established assessment formats. In physics education, where problem solving plays a central role in instruction and assessment, it is therefore essential to understand the physics-specific problem-solving capabilities of LLMs. Such understanding is key to informing responsible and pedagogically sound approaches to integrating LLMs into instruction and assessment. This study therefore compares the problem-solving performance of a general-purpose LLM (GPT-4o, using varying prompting techniques) and a reasoning-optimized model (o1-preview) with that of participants of the German Physics Olympiad, based on a set of well-defined Olympiad problems. In addition to evaluating the correctness of the generated solutions, the study analyzes characteristic strengths and limitations of LLM-generated solutions. The findings of this study indicate that both tested LLMs (GPT-4o and o1-preview) demonstrate advanced problem-solving capabilities on Olympiad-type physics problems, on average outperforming the human participants. Prompting techniques had little effect on GPT-4o's performance, while o1-preview almost consistently outperformed both GPT-4o and the human benchmark. Based on these findings, the study discusses implications for the design of summative and formative assessment in physics education, including how to uphold assessment integrity and support students in critically engaging with LLMs. 

---
# Air-Ground Collaboration for Language-Specified Missions in Unknown Environments 

**Authors**: Fernando Cladera, Zachary Ravichandran, Jason Hughes, Varun Murali, Carlos Nieto-Granda, M. Ani Hsieh, George J. Pappas, Camillo J. Taylor, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.09108)  

**Abstract**: As autonomous robotic systems become increasingly mature, users will want to specify missions at the level of intent rather than in low-level detail. Language is an expressive and intuitive medium for such mission specification. However, realizing language-guided robotic teams requires overcoming significant technical hurdles. Interpreting and realizing language-specified missions requires advanced semantic reasoning. Successful heterogeneous robots must effectively coordinate actions and share information across varying viewpoints. Additionally, communication between robots is typically intermittent, necessitating robust strategies that leverage communication opportunities to maintain coordination and achieve mission objectives. In this work, we present a first-of-its-kind system where an unmanned aerial vehicle (UAV) and an unmanned ground vehicle (UGV) are able to collaboratively accomplish missions specified in natural language while reacting to changes in specification on the fly. We leverage a Large Language Model (LLM)-enabled planner to reason over semantic-metric maps that are built online and opportunistically shared between an aerial and a ground robot. We consider task-driven navigation in urban and rural areas. Our system must infer mission-relevant semantics and actively acquire information via semantic mapping. In both ground and air-ground teaming experiments, we demonstrate our system on seven different natural-language specifications at up to kilometer-scale navigation. 

---
# SALM: A Multi-Agent Framework for Language Model-Driven Social Network Simulation 

**Authors**: Gaurav Koley  

**Link**: [PDF](https://arxiv.org/pdf/2505.09081)  

**Abstract**: Contemporary approaches to agent-based modeling (ABM) of social systems have traditionally emphasized rule-based behaviors, limiting their ability to capture nuanced dynamics by moving beyond predefined rules and leveraging contextual understanding from LMs of human social interaction. This paper presents SALM (Social Agent LM Framework), a novel approach for integrating language models (LMs) into social network simulation that achieves unprecedented temporal stability in multi-agent scenarios. Our primary contributions include: (1) a hierarchical prompting architecture enabling stable simulation beyond 4,000 timesteps while reducing token usage by 73%, (2) an attention-based memory system achieving 80% cache hit rates (95% CI [78%, 82%]) with sub-linear memory growth of 9.5%, and (3) formal bounds on personality stability. Through extensive validation against SNAP ego networks, we demonstrate the first LLM-based framework capable of modeling long-term social phenomena while maintaining empirically validated behavioral fidelity. 

---
# Variational Prefix Tuning for Diverse and Accurate Code Summarization Using Pre-trained Language Models 

**Authors**: Junda Zhao, Yuliang Song, Eldan Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2505.09062)  

**Abstract**: Recent advancements in source code summarization have leveraged transformer-based pre-trained models, including Large Language Models of Code (LLMCs), to automate and improve the generation of code summaries. However, existing methods often focus on generating a single high-quality summary for a given source code, neglecting scenarios where the generated summary might be inadequate and alternative options are needed. In this paper, we introduce Variational Prefix Tuning (VPT), a novel approach that enhances pre-trained models' ability to generate diverse yet accurate sets of summaries, allowing the user to choose the most suitable one for the given source code. Our method integrates a Conditional Variational Autoencoder (CVAE) framework as a modular component into pre-trained models, enabling us to model the distribution of observed target summaries and sample continuous embeddings to be used as prefixes to steer the generation of diverse outputs during decoding. Importantly, we construct our method in a parameter-efficient manner, eliminating the need for expensive model retraining, especially when using LLMCs. Furthermore, we employ a bi-criteria reranking method to select a subset of generated summaries, optimizing both the diversity and the accuracy of the options presented to users. We present extensive experimental evaluations using widely used datasets and current state-of-the-art pre-trained code summarization models to demonstrate the effectiveness of our approach and its adaptability across models. 

---
# Tests as Prompt: A Test-Driven-Development Benchmark for LLM Code Generation 

**Authors**: Yi Cui  

**Link**: [PDF](https://arxiv.org/pdf/2505.09027)  

**Abstract**: We introduce WebApp1K, a novel benchmark for evaluating large language models (LLMs) in test-driven development (TDD) tasks, where test cases serve as both prompt and verification for code generation. Unlike traditional approaches relying on natural language prompts, our benchmark emphasizes the ability of LLMs to interpret and implement functionality directly from test cases, reflecting real-world software development practices. Comprising 1000 diverse challenges across 20 application domains, the benchmark evaluates LLMs on their ability to generate compact, functional code under the constraints of context length and multi-feature complexity. Our findings highlight instruction following and in-context learning as critical capabilities for TDD success, surpassing the importance of general coding proficiency or pretraining knowledge. Through comprehensive evaluation of 19 frontier models, we reveal performance bottlenecks, such as instruction loss in long prompts, and provide a detailed error analysis spanning multiple root causes. This work underscores the practical value of TDD-specific benchmarks and lays the foundation for advancing LLM capabilities in rigorous, application-driven coding scenarios. 

---
# AI-Mediated Code Comment Improvement 

**Authors**: Maria Dhakal, Chia-Yi Su, Robert Wallace, Chris Fakhimi, Aakash Bansal, Toby Li, Yu Huang, Collin McMillan  

**Link**: [PDF](https://arxiv.org/pdf/2505.09021)  

**Abstract**: This paper describes an approach to improve code comments along different quality axes by rewriting those comments with customized Artificial Intelligence (AI)-based tools. We conduct an empirical study followed by grounded theory qualitative analysis to determine the quality axes to improve. Then we propose a procedure using a Large Language Model (LLM) to rewrite existing code comments along the quality axes. We implement our procedure using GPT-4o, then distil the results into a smaller model capable of being run in-house, so users can maintain data custody. We evaluate both our approach using GPT-4o and the distilled model versions. We show in an evaluation how our procedure improves code comments along the quality axes. We release all data and source code in an online repository for reproducibility. 

---
# Human-like Cognitive Generalization for Large Models via Brain-in-the-loop Supervision 

**Authors**: Jiaxuan Chen, Yu Qi, Yueming Wang, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.09085)  

**Abstract**: Recent advancements in deep neural networks (DNNs), particularly large-scale language models, have demonstrated remarkable capabilities in image and natural language understanding. Although scaling up model parameters with increasing volume of training data has progressively improved DNN capabilities, achieving complex cognitive abilities - such as understanding abstract concepts, reasoning, and adapting to novel scenarios, which are intrinsic to human cognition - remains a major challenge. In this study, we show that brain-in-the-loop supervised learning, utilizing a small set of brain signals, can effectively transfer human conceptual structures to DNNs, significantly enhancing their comprehension of abstract and even unseen concepts. Experimental results further indicate that the enhanced cognitive capabilities lead to substantial performance gains in challenging tasks, including few-shot/zero-shot learning and out-of-distribution recognition, while also yielding highly interpretable concept representations. These findings highlight that human-in-the-loop supervision can effectively augment the complex cognitive abilities of large models, offering a promising pathway toward developing more human-like cognitive abilities in artificial systems. 

---
# Generative AI for Autonomous Driving: Frontiers and Opportunities 

**Authors**: Yuping Wang, Shuo Xing, Cui Can, Renjie Li, Hongyuan Hua, Kexin Tian, Zhaobin Mo, Xiangbo Gao, Keshu Wu, Sulong Zhou, Hengxu You, Juntong Peng, Junge Zhang, Zehao Wang, Rui Song, Mingxuan Yan, Walter Zimmer, Xingcheng Zhou, Peiran Li, Zhaohan Lu, Chia-Ju Chen, Yue Huang, Ryan A. Rossi, Lichao Sun, Hongkai Yu, Zhiwen Fan, Frank Hao Yang, Yuhao Kang, Ross Greer, Chenxi Liu, Eun Hak Lee, Xuan Di, Xinyue Ye, Liu Ren, Alois Knoll, Xiaopeng Li, Shuiwang Ji, Masayoshi Tomizuka, Marco Pavone, Tianbao Yang, Jing Du, Ming-Hsuan Yang, Hua Wei, Ziran Wang, Yang Zhou, Jiachen Li, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08854)  

**Abstract**: Generative Artificial Intelligence (GenAI) constitutes a transformative technological wave that reconfigures industries through its unparalleled capabilities for content creation, reasoning, planning, and multimodal understanding. This revolutionary force offers the most promising path yet toward solving one of engineering's grandest challenges: achieving reliable, fully autonomous driving, particularly the pursuit of Level 5 autonomy. This survey delivers a comprehensive and critical synthesis of the emerging role of GenAI across the autonomous driving stack. We begin by distilling the principles and trade-offs of modern generative modeling, encompassing VAEs, GANs, Diffusion Models, and Large Language Models (LLMs). We then map their frontier applications in image, LiDAR, trajectory, occupancy, video generation as well as LLM-guided reasoning and decision making. We categorize practical applications, such as synthetic data workflows, end-to-end driving strategies, high-fidelity digital twin systems, smart transportation networks, and cross-domain transfer to embodied AI. We identify key obstacles and possibilities such as comprehensive generalization across rare cases, evaluation and safety checks, budget-limited implementation, regulatory compliance, ethical concerns, and environmental effects, while proposing research plans across theoretical assurances, trust metrics, transport integration, and socio-technical influence. By unifying these threads, the survey provides a forward-looking reference for researchers, engineers, and policymakers navigating the convergence of generative AI and advanced autonomous mobility. An actively maintained repository of cited works is available at this https URL. 

---
# CellTypeAgent: Trustworthy cell type annotation with Large Language Models 

**Authors**: Jiawen Chen, Jianghao Zhang, Huaxiu Yao, Yun Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.08844)  

**Abstract**: Cell type annotation is a critical yet laborious step in single-cell RNA sequencing analysis. We present a trustworthy large language model (LLM)-agent, CellTypeAgent, which integrates LLMs with verification from relevant databases. CellTypeAgent achieves higher accuracy than existing methods while mitigating hallucinations. We evaluated CellTypeAgent across nine real datasets involving 303 cell types from 36 tissues. This combined approach holds promise for more efficient and reliable cell type annotation. 

---
# Federated Large Language Models: Feasibility, Robustness, Security and Future Directions 

**Authors**: Wenhao Jiang, Yuchuan Luo, Guilin Deng, Silong Chen, Xu Yang, Shihong Wu, Xinwen Gao, Lin Liu, Shaojing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08830)  

**Abstract**: The integration of Large Language Models (LLMs) and Federated Learning (FL) presents a promising solution for joint training on distributed data while preserving privacy and addressing data silo issues. However, this emerging field, known as Federated Large Language Models (FLLM), faces significant challenges, including communication and computation overheads, heterogeneity, privacy and security concerns. Current research has primarily focused on the feasibility of FLLM, but future trends are expected to emphasize enhancing system robustness and security. This paper provides a comprehensive review of the latest advancements in FLLM, examining challenges from four critical perspectives: feasibility, robustness, security, and future directions. We present an exhaustive survey of existing studies on FLLM feasibility, introduce methods to enhance robustness in the face of resource, data, and task heterogeneity, and analyze novel risks associated with this integration, including privacy threats and security challenges. We also review the latest developments in defense mechanisms and explore promising future research directions, such as few-shot learning, machine unlearning, and IP protection. This survey highlights the pressing need for further research to enhance system robustness and security while addressing the unique challenges posed by the integration of FL and LLM. 

---
# Self Rewarding Self Improving 

**Authors**: Toby Simonds, Kevin Lopez, Akira Yoshiyama, Dominique Garmier  

**Link**: [PDF](https://arxiv.org/pdf/2505.08827)  

**Abstract**: We demonstrate that large language models can effectively self-improve through self-judging without requiring reference solutions, leveraging the inherent asymmetry between generating and verifying solutions. Our experiments on Countdown puzzles and MIT Integration Bee problems show that models can provide reliable reward signals without ground truth answers, enabling reinforcement learning in domains previously not possible. By implementing self-judging, we achieve significant performance gains maintaining alignment with formal verification. When combined with synthetic question generation, we establish a complete self-improvement loop where models generate practice problems, solve them, and evaluate their own performance-achieving an 8% improvement with Qwen 2.5 7B over baseline and surpassing GPT-4o performance on integration tasks. Our findings demonstrate that LLM judges can provide effective reward signals for training models, unlocking many reinforcement learning environments previously limited by the difficulty of creating programmatic rewards. This suggests a potential paradigm shift toward AI systems that continuously improve through self-directed learning rather than human-guided training, potentially accelerating progress in domains with scarce training data or complex evaluation requirements. 

---
