# Resource Allocation in Recommender Systems for Global KPI Improvement

## Contributors
Riccardo Galanti
Alessandro Padella

Process-aware Recommender systems are information systems designed to monitor the execution of processes, predict their outcomes, and suggest effective interventions to achieve better results, with respect to reference KPIs (Key Performance Indicators).  
Interventions typically consist of suggesting an activity to be assigned to a certain resource. 
State of the art typically proposes interventions for single cases in isolation. However, since resources are shared among cases, this might impact the effectiveness of the available interventions for other cases that would require one. As result, the overall KPI improvement is partially hampered. 
This paper proposes an approach to assign resources to needed cases, aiming to improve the overall KPI values for all cases together, namely the summation of KPI values for all cases. Experiments conducted on two real-life case studies illustrate that globally considering all needing cases together allows a better global KPI improvement, compared with a more greedy approach where interventions are proposed one after the other.

Requirements: req.txt

Training framework for CatBoost predictor and Building a transition system: main_recsys.py

Generation of DeltaRank: main_global_p.py

Generation of Profiles Rank: perturbations.py 

Evaluation of system: evaluate_generated_set.py

Datasets are bigger than 25MB, so they are both available emailing alessandro.padella@phd.unipd.it. VINST is also available following the link on the paper.
