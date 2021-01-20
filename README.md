# IFCB
The implementation for paper IoTs Fog-Cloud combining with Blockchain technology

# How to run
```code 
1. main_single_objective.py
- For single objective only

2. test_nsga_ii.py
- For pareto-front (multi-objective)

3. data_generator.py
- Generate fog-cloud-blockchain architecture.

4. config.py
- Config for the whole project



```

# Notes

```code 
- Cloud latency stats:
    http://cloudharmony.com/speedtest-for-aws:ec2
- Cloud cost stats:
    https://www.datamation.com/cloud-computing/cloud-costs.html
    
Computation: 0.07$ - 1 hour - 2GB RAM 
==> 1s -> 3600 * 2 * 1000 000 000 = 7200 000 000 000 - 0.07$
==> 1 Byte: 9.7e-15$ 

Storage: 0.07$ - 1 month - 1GB Disk
==> 1s -> 30 * 86400 * 1000 000 000 = 2592 000 000 000 000 - 0.07$
==> 1 Byte: 2.7e-17 

- punishment function: based on time-delay of each single task
https://www.researchgate.net/publication/267844406_Niche_Construction_Sustainability_and_Evolutionary_Ecology_of_Cancer

fx  = 0 if delay_time < 0
    = delay_time if 0 < delay_time < 1
    = (delay_time**2 + 1)/2   

- Inverse-time function: based on living-time (tau) of data in fog
- The function in NCA paper is exponential of 2*(t-j). It is not a good way in computing especially with large tasks
 and small fogs.  (For example: 1000 tasks - 10 fogs --> some fogs with 2**100 will appear.
fx = 1 / ((t-j)**2 + 1)


- A framework to simulate cloud-fog in Python, but in their examples there is no examples of how fog-cloud simulation
 is carry out.
https://www.researchgate.net/post/Is-there-any-Python-Cloud-Computing-Simulator-with-Autoscaling-Features
https://github.com/acsicuib/YAFS/tree/YAFS3
https://yafs.readthedocs.io/en/latest/examples/tutorial_example.html


- Some citing papers:
    + #tasks is small and each task with small size:
        https://sci-hub.se/10.1109/iwcmc.2019.8766437
        http://www.es.mdh.se/pdf_publications/5957.pdf
    + Figure the request from end user to fog-cloud:
        https://www.researchgate.net/publication/335359931_A_Method_Based_on_the_Combination_of_Laxity_and_Ant_Colony_System_for_Cloud-Fog_Task_Scheduling/figures?lo=1
    + Priority task in scheduling algorithm
        https://core.ac.uk/download/pdf/159815677.pdf
    https://sci-hub.se/10.1109/tcc.2013.2



- Fog-Cloud-Peer nodes:
    +) 3f-1: 2-fog, 8-cloud, 5-peer
    +) 3f+1: 4-fog, 10-cloud, 7-peer


- Testing with 1000 tasks: (Assumption with 2 clouds, 8 fogs and 5 peers) (3f-1)
    + Number of variables (dimensions) = (2+8)*1000 = 10000 --> Can't do with Metaheuristic Algorithms
    + 1 epoch: 400 - 500 seconds ----> 1000 epochs: 400000 - 500000 seconds --> 1 Algorithm with 1 trial: 4-6 days 
--> So maximum tasks should be: 500 or less
    
- Testing with 500 tasks 
    + #dim = 10 * 500 = 5000
    + 1 epoch: 50 - 100 seconds ---> 1000 epochs: 50000 - 100000 seconds --> 1 Algo with 1 trial: 14-hour to 28 hours




```