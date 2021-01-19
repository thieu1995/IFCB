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