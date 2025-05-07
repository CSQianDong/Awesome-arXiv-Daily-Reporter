# RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation 

**Authors**: Tiantian Gan, Qiyao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.03275)  

**Abstract**: Large language models (LLMs) struggle to effectively utilize a growing number of external tools, such as those defined by the Model Context Protocol (MCP)\cite{IntroducingMCP}, due to prompt bloat and selection complexity. We introduce RAG-MCP, a Retrieval-Augmented Generation framework that overcomes this challenge by offloading tool discovery. RAG-MCP uses semantic retrieval to identify the most relevant MCP(s) for a given query from an external index before engaging the LLM. Only the selected tool descriptions are passed to the model, drastically reducing prompt size and simplifying decision-making. Experiments, including an MCP stress test, demonstrate RAG-MCP significantly cuts prompt tokens (e.g., by over 50%) and more than triples tool selection accuracy (43.13% vs 13.62% baseline) on benchmark tasks. RAG-MCP enables scalable and accurate tool integration for LLMs. 

---
# Capability-Driven Skill Generation with LLMs: A RAG-Based Approach for Reusing Existing Libraries and Interfaces 

**Authors**: Luis Miguel Vieira da Silva, Aljosha Köcher, Nicolas König, Felix Gehlhoff, Alexander Fay  

**Link**: [PDF](https://arxiv.org/pdf/2505.03295)  

**Abstract**: Modern automation systems increasingly rely on modular architectures, with capabilities and skills as one solution approach. Capabilities define the functions of resources in a machine-readable form and skills provide the concrete implementations that realize those capabilities. However, the development of a skill implementation conforming to a corresponding capability remains a time-consuming and challenging task. In this paper, we present a method that treats capabilities as contracts for skill implementations and leverages large language models to generate executable code based on natural language user input. A key feature of our approach is the integration of existing software libraries and interface technologies, enabling the generation of skill implementations across different target languages. We introduce a framework that allows users to incorporate their own libraries and resource interfaces into the code generation process through a retrieval-augmented generation architecture. The proposed method is evaluated using an autonomous mobile robot controlled via Python and ROS 2, demonstrating the feasibility and flexibility of the approach. 

---
# An Analysis of Hyper-Parameter Optimization Methods for Retrieval Augmented Generation 

**Authors**: Matan Orbach, Ohad Eytan, Benjamin Sznajder, Ariel Gera, Odellia Boni, Yoav Kantor, Gal Bloch, Omri Levy, Hadas Abraham, Nitzan Barzilay, Eyal Shnarch, Michael E. Factor, Shila Ofek-Koifman, Paula Ta-Shma, Assaf Toledo  

**Link**: [PDF](https://arxiv.org/pdf/2505.03452)  

**Abstract**: Finding the optimal Retrieval-Augmented Generation (RAG) configuration for a given use case can be complex and expensive. Motivated by this challenge, frameworks for RAG hyper-parameter optimization (HPO) have recently emerged, yet their effectiveness has not been rigorously benchmarked. To address this gap, we present a comprehensive study involving 5 HPO algorithms over 5 datasets from diverse domains, including a new one collected for this work on real-world product documentation. Our study explores the largest HPO search space considered to date, with two optimized evaluation metrics. Analysis of the results shows that RAG HPO can be done efficiently, either greedily or with iterative random search, and that it significantly boosts RAG performance for all datasets. For greedy HPO approaches, we show that optimizing models first is preferable to the prevalent practice of optimizing sequentially according to the RAG pipeline order. 

---
# Lightweight Clinical Decision Support System using QLoRA-Fine-Tuned LLMs and Retrieval-Augmented Generation 

**Authors**: Mohammad Shoaib Ansari, Mohd Sohail Ali Khan, Shubham Revankar, Aditya Varma, Anil S. Mokhade  

**Link**: [PDF](https://arxiv.org/pdf/2505.03406)  

**Abstract**: This research paper investigates the application of Large Language Models (LLMs) in healthcare, specifically focusing on enhancing medical decision support through Retrieval-Augmented Generation (RAG) integrated with hospital-specific data and fine-tuning using Quantized Low-Rank Adaptation (QLoRA). The system utilizes Llama 3.2-3B-Instruct as its foundation model. By embedding and retrieving context-relevant healthcare information, the system significantly improves response accuracy. QLoRA facilitates notable parameter efficiency and memory optimization, preserving the integrity of medical information through specialized quantization techniques. Our research also shows that our model performs relatively well on various medical benchmarks, indicating that it can be used to make basic medical suggestions. This paper details the system's technical components, including its architecture, quantization methods, and key healthcare applications such as enhanced disease prediction from patient symptoms and medical history, treatment suggestions, and efficient summarization of complex medical reports. We touch on the ethical considerations-patient privacy, data security, and the need for rigorous clinical validation-as well as the practical challenges of integrating such systems into real-world healthcare workflows. Furthermore, the lightweight quantized weights ensure scalability and ease of deployment even in low-resource hospital environments. Finally, the paper concludes with an analysis of the broader impact of LLMs on healthcare and outlines future directions for LLMs in medical settings. 

---
# Enhancing tutoring systems by leveraging tailored promptings and domain knowledge with Large Language Models 

**Authors**: Mohsen Balavar, Wenli Yang, David Herbert, Soonja Yeom  

**Link**: [PDF](https://arxiv.org/pdf/2505.02849)  

**Abstract**: Recent advancements in artificial intelligence (AI) and machine learning have reignited interest in their impact on Computer-based Learning (CBL). AI-driven tools like ChatGPT and Intelligent Tutoring Systems (ITS) have enhanced learning experiences through personalisation and flexibility. ITSs can adapt to individual learning needs and provide customised feedback based on a student's performance, cognitive state, and learning path. Despite these advances, challenges remain in accommodating diverse learning styles and delivering real-time, context-aware feedback. Our research aims to address these gaps by integrating skill-aligned feedback via Retrieval Augmented Generation (RAG) into prompt engineering for Large Language Models (LLMs) and developing an application to enhance learning through personalised tutoring in a computer science programming context. The pilot study evaluated a proposed system using three quantitative metrics: readability score, response time, and feedback depth, across three programming tasks of varying complexity. The system successfully sorted simulated students into three skill-level categories and provided context-aware feedback. This targeted approach demonstrated better effectiveness and adaptability compared to general methods. 

---
# Direct Retrieval-augmented Optimization: Synergizing Knowledge Selection and Language Models 

**Authors**: Zhengliang Shi, Lingyong Yan, Weiwei Sun, Yue Feng, Pengjie Ren, Xinyu Ma, Shuaiqiang Wang, Dawei Yin, Maarten de Rijke, Zhaochun Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.03075)  

**Abstract**: Retrieval-augmented generation (RAG) integrates large language models ( LLM s) with retrievers to access external knowledge, improving the factuality of LLM generation in knowledge-grounded tasks. To optimize the RAG performance, most previous work independently fine-tunes the retriever to adapt to frozen LLM s or trains the LLMs to use documents retrieved by off-the-shelf retrievers, lacking end-to-end training supervision. Recent work addresses this limitation by jointly training these two components but relies on overly simplifying assumptions of document independence, which has been criticized for being far from real-world scenarios. Thus, effectively optimizing the overall RAG performance remains a critical challenge.
We propose a direct retrieval-augmented optimization framework, named DRO, that enables end-to-end training of two key components: (i) a generative knowledge selection model and (ii) an LLM generator. DRO alternates between two phases: (i) document permutation estimation and (ii) re-weighted maximization, progressively improving RAG components through a variational approach. In the estimation step, we treat document permutation as a latent variable and directly estimate its distribution from the selection model by applying an importance sampling strategy. In the maximization step, we calibrate the optimization expectation using importance weights and jointly train the selection model and LLM generator. Our theoretical analysis reveals that DRO is analogous to policy-gradient methods in reinforcement learning. Extensive experiments conducted on five datasets illustrate that DRO outperforms the best baseline with 5%-15% improvements in EM and F1. We also provide in-depth experiments to qualitatively analyze the stability, convergence, and variance of DRO. 

---
