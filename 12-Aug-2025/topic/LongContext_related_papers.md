# Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL 

**Authors**: Jiaxuan Gao, Wei Fu, Minyang Xie, Shusheng Xu, Chuyi He, Zhiyu Mei, Banghua Zhu, Yi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.07976)  

**Abstract**: Recent advancements in LLM-based agents have demonstrated remarkable capabilities in handling complex, knowledge-intensive tasks by integrating external tools. Among diverse choices of tools, search tools play a pivotal role in accessing vast external knowledge. However, open-source agents still fall short of achieving expert-level Search Intelligence, the ability to resolve ambiguous queries, generate precise searches, analyze results, and conduct thorough exploration. Existing approaches fall short in scalability, efficiency, and data quality. For example, small turn limits in existing online RL methods, e.g. <=10, restrict complex strategy learning. This paper introduces ASearcher, an open-source project for large-scale RL training of search agents. Our key contributions include: (1) Scalable fully asynchronous RL training that enables long-horizon search while maintaining high training efficiency. (2) A prompt-based LLM agent that autonomously synthesizes high-quality and challenging QAs, creating a large-scale QA dataset. Through RL training, our prompt-based QwQ-32B agent achieves substantial improvements, with 46.7% and 20.8% Avg@4 gains on xBench and GAIA, respectively. Notably, our agent exhibits extreme long-horizon search, with tool calls exceeding 40 turns and output tokens exceeding 150k during training time. With a simple agent design and no external LLMs, ASearcher-Web-QwQ achieves Avg@4 scores of 42.1 on xBench and 52.8 on GAIA, surpassing existing open-source 32B agents. We open-source our models, training data, and codes in this https URL. 

---
# Positional Biases Shift as Inputs Approach Context Window Limits 

**Authors**: Blerta Veseli, Julian Chibane, Mariya Toneva, Alexander Koller  

**Link**: [PDF](https://arxiv.org/pdf/2508.07479)  

**Abstract**: Large Language Models (LLMs) often struggle to use information across long inputs effectively. Prior work has identified positional biases, such as the Lost in the Middle (LiM) effect, where models perform better when information appears at the beginning (primacy bias) or end (recency bias) of the input, rather than in the middle. However, long-context studies have not consistently replicated these effects, raising questions about their intensity and the conditions under which they manifest. To address this, we conducted a comprehensive analysis using relative rather than absolute input lengths, defined with respect to each model's context window. Our findings reveal that the LiM effect is strongest when inputs occupy up to 50% of a model's context window. Beyond that, the primacy bias weakens, while recency bias remains relatively stable. This effectively eliminates the LiM effect; instead, we observe a distance-based bias, where model performance is better when relevant information is closer to the end of the input. Furthermore, our results suggest that successful retrieval is a prerequisite for reasoning in LLMs, and that the observed positional biases in reasoning are largely inherited from retrieval. These insights have implications for long-context tasks, the design of future LLM benchmarks, and evaluation methodologies for LLMs handling extended inputs. 

---
# Many-Turn Jailbreaking 

**Authors**: Xianjun Yang, Liqiang Xiao, Shiyang Li, Faisal Ladhak, Hyokun Yun, Linda Ruth Petzold, Yi Xu, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06755)  

**Abstract**: Current jailbreaking work on large language models (LLMs) aims to elicit unsafe outputs from given prompts. However, it only focuses on single-turn jailbreaking targeting one specific query. On the contrary, the advanced LLMs are designed to handle extremely long contexts and can thus conduct multi-turn conversations. So, we propose exploring multi-turn jailbreaking, in which the jailbroken LLMs are continuously tested on more than the first-turn conversation or a single target query. This is an even more serious threat because 1) it is common for users to continue asking relevant follow-up questions to clarify certain jailbroken details, and 2) it is also possible that the initial round of jailbreaking causes the LLMs to respond to additional irrelevant questions consistently. As the first step (First draft done at June 2024) in exploring multi-turn jailbreaking, we construct a Multi-Turn Jailbreak Benchmark (MTJ-Bench) for benchmarking this setting on a series of open- and closed-source models and provide novel insights into this new safety threat. By revealing this new vulnerability, we aim to call for community efforts to build safer LLMs and pave the way for a more in-depth understanding of jailbreaking LLMs. 

---
# MuaLLM: A Multimodal Large Language Model Agent for Circuit Design Assistance with Hybrid Contextual Retrieval-Augmented Generation 

**Authors**: Pravallika Abbineni, Saoud Aldowaish, Colin Liechty, Soroosh Noorzad, Ali Ghazizadeh, Morteza Fayazi  

**Link**: [PDF](https://arxiv.org/pdf/2508.08137)  

**Abstract**: Conducting a comprehensive literature review is crucial for advancing circuit design methodologies. However, the rapid influx of state-of-the-art research, inconsistent data representation, and the complexity of optimizing circuit design objectives make this task significantly challenging. In this paper, we propose MuaLLM, an open-source multimodal Large Language Model (LLM) agent for circuit design assistance that integrates a hybrid Retrieval-Augmented Generation (RAG) framework with an adaptive vector database of circuit design research papers. Unlike conventional LLMs, the MuaLLM agent employs a Reason + Act (ReAct) workflow for iterative reasoning, goal-setting, and multi-step information retrieval. It functions as a question-answering design assistant, capable of interpreting complex queries and providing reasoned responses grounded in circuit literature. Its multimodal capabilities enable processing of both textual and visual data, facilitating more efficient and comprehensive analysis. The system dynamically adapts using intelligent search tools, automated document retrieval from the internet, and real-time database updates. Unlike conventional approaches constrained by model context limits, MuaLLM decouples retrieval from inference, enabling scalable reasoning over arbitrarily large corpora. At the maximum context length supported by standard LLMs, MuaLLM remains up to 10x less costly and 1.6x faster while maintaining the same accuracy. This allows rapid, no-human-in-the-loop database generation, overcoming the bottleneck of simulation-based dataset creation for circuits. To evaluate MuaLLM, we introduce two custom datasets: RAG-250, targeting retrieval and citation performance, and Reasoning-100 (Reas-100), focused on multistep reasoning in circuit design. MuaLLM achieves 90.1% recall on RAG-250, and 86.8% accuracy on Reas-100. 

---
