# Safeguarding RAG Pipelines with GMTP: A Gradient-based Masked Token Probability Method for Poisoned Document Detection 

**Authors**: San Kim, Jonghwi Kim, Yejin Jeon, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.18202)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by providing external knowledge for accurate and up-to-date responses. However, this reliance on external sources exposes a security risk, attackers can inject poisoned documents into the knowledge base to steer the generation process toward harmful or misleading outputs. In this paper, we propose Gradient-based Masked Token Probability (GMTP), a novel defense method to detect and filter out adversarially crafted documents. Specifically, GMTP identifies high-impact tokens by examining gradients of the retriever's similarity function. These key tokens are then masked, and their probabilities are checked via a Masked Language Model (MLM). Since injected tokens typically exhibit markedly low masked-token probabilities, this enables GMTP to easily detect malicious documents and achieve high-precision filtering. Experiments demonstrate that GMTP is able to eliminate over 90% of poisoned content while retaining relevant documents, thus maintaining robust retrieval and generation performance across diverse datasets and adversarial settings. 

---
# VERIRAG: Healthcare Claim Verification via Statistical Audit in Retrieval-Augmented Generation 

**Authors**: Shubham Mohole, Hongjun Choi, Shusen Liu, Christine Klymko, Shashank Kushwaha, Derek Shi, Wesam Sakla, Sainyam Galhotra, Ruben Glatt  

**Link**: [PDF](https://arxiv.org/pdf/2507.17948)  

**Abstract**: Retrieval-augmented generation (RAG) systems are increasingly adopted in clinical decision support, yet they remain methodologically blind-they retrieve evidence but cannot vet its scientific quality. A paper claiming "Antioxidant proteins decreased after alloferon treatment" and a rigorous multi-laboratory replication study will be treated as equally credible, even if the former lacked scientific rigor or was even retracted. To address this challenge, we introduce VERIRAG, a framework that makes three notable contributions: (i) the Veritable, an 11-point checklist that evaluates each source for methodological rigor, including data integrity and statistical validity; (ii) a Hard-to-Vary (HV) Score, a quantitative aggregator that weights evidence by its quality and diversity; and (iii) a Dynamic Acceptance Threshold, which calibrates the required evidence based on how extraordinary a claim is. Across four datasets-comprising retracted, conflicting, comprehensive, and settled science corpora-the VERIRAG approach consistently outperforms all baselines, achieving absolute F1 scores ranging from 0.53 to 0.65, representing a 10 to 14 point improvement over the next-best method in each respective dataset. We will release all materials necessary for reproducing our results. 

---
# GenAI for Automotive Software Development: From Requirements to Wheels 

**Authors**: Nenad Petrovic, Fengjunjie Pan, Vahid Zolfaghari, Krzysztof Lebioda, Andre Schamschurko, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2507.18223)  

**Abstract**: This paper introduces a GenAI-empowered approach to automated development of automotive software, with emphasis on autonomous and Advanced Driver Assistance Systems (ADAS) capabilities. The process starts with requirements as input, while the main generated outputs are test scenario code for simulation environment, together with implementation of desired ADAS capabilities targeting hardware platform of the vehicle connected to testbench. Moreover, we introduce additional steps for requirements consistency checking leveraging Model-Driven Engineering (MDE). In the proposed workflow, Large Language Models (LLMs) are used for model-based summarization of requirements (Ecore metamodel, XMI model instance and OCL constraint creation), test scenario generation, simulation code (Python) and target platform code generation (C++). Additionally, Retrieval Augmented Generation (RAG) is adopted to enhance test scenario generation from autonomous driving regulations-related documents. Our approach aims shorter compliance and re-engineering cycles, as well as reduced development and testing time when it comes to ADAS-related capabilities. 

---
