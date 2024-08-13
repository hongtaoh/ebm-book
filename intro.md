# Event-Based Model (EBM)

Event-based Model (EBM) is used to model disease progression. We can obtain the estimated order where different biological factors get affected by a specific disease. These factors are called "biomarkers". The order contains multiple **stages**. The biomarker data come from patients' visits where they typically go through neuropsych (e.g., MMSE) and/or biological examiations (e.g., blood pressure).  Visits data do not have to be longitudinal; they can be single visits from a cohort of patients. 

We have several assumptions in EBM:
- The disease is irreversible (i.e., a patient cannot go from stage 2 to stage 1)
- Biomarkers are independent; i.e., we cannot infer information about one biomarker from that about another one. 
- The order in which different biomarkers get affected by the disease is the same across all patients.

This book contains chapters that explain the how to compuate the likelihood of biomarker values and how to generate biomarker data through simulations. 

- [Likelihood of Biomarker Data](./distributions.ipynb)

- [Generative Process](./generative_process.ipynb)

- [Estimating theta and phi](./estimate_one.ipynb)

- [Estimating k_j](./estimate_two.ipynb)

- [Estimating biomarker ordering](./estimate_three.ipynb)

- [Research Plan](./plan.md)