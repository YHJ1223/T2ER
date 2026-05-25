# T2ER: A Chinese Benchmark for Text-to-Executable-Rule Generation in Medical Insurance


## 📖 Abstract

Medical insurance plays a pivotal role in the healthcare system, where effective supervision relies on the precise execution of complex Medical Insurance Rules (MIR). Existing solutions for medical insurance violation detection mainly rely on costly manual labor, resulting in limited efficiency. Although current LLMs have demonstrated significant potential in general NLP tasks, they still struggle to understand MIRs, and a corresponding evaluation benchmark is lacking. Hence, we propose MIR-Bench, a benchmark to evaluate the capability of LLMs in converting natural language MIR texts into executable rule functions. MIR-Bench consists of 2,115 MIRs and 5,626 patient test cases for execution verification. Furthermore, we design a dual evaluation framework spanning rule and execution levels to comprehensively assess transformation quality. We conducted extensive evaluations on advanced LLMs. The results reveal a significant human-AI performance gap, indicating that MIR conversion remains a formidable challenge.

![Benchmark Construction Pipeline](assets/pipeline.png)



## Ethical Consideration

Our benchmark raises no privacy concerns. All rule texts are extracted from publicly released medical insurance policy documents, and all EMRs are fully synthetic and contain no personally identifiable information. Therefore, MIR-Bench does not involve real individuals, sensitive attributes, or unauthorized data use, and we are not aware of other ethical issues arising from the dataset.
