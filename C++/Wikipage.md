# Random-Key Optimizer (RKO)

The Random-Key Optimizer (RKO) is a metaheuristic optimization algorithm that employs a continuous random-key representation to encode solutions. It is designed to solve combinatorial and continuous optimization problems by leveraging evolutionary and trajectory algorithm principles.

## Overview
RKO is based on the concept of random-key encoding, where solutions are represented as vectors of real numbers within a predefined range (typically [0,1]). These vectors are then decoded into feasible solutions using a problem-specific decoding function. The method is inspired by the Biased Random-Key Genetic Algorithm (BRKGA) but incorporates modifications to enhance efficiency and adaptability.

## History
The RKO method was introduced as a general-purpose optimization framework for solving complex decision-making problems. It builds upon the foundation of random-key genetic algorithms (RKGA), first introduced in the late 1990s, and extends their applicability by incorporating adaptive strategies and heuristic refinements.

Literature review about RKO:
- Bean (1994): Random-Key Genetic Algorithms (RKGA)
- Gonçalves and Almeida (2002), Gonçalves & Resende (2011): Biased Random-Key Genetic Algorithms (BRKGA)
- Lin et al. (2010); Bewoor et al. (2017, 2018): Particle Swarm Optimization (PSO)
- Garcia-Santiago et al. (2015): Harmony Search (HS)
- Pessoa and Andrade (2018): Iterated Local Search (ILS), Iterated Greedy Search (IGS)
- Andrade et al. (2019): ILS, Tabu Search (TS) and Simulated Annealing (SA)
- Andrade et al. (2021): Implicit Path-Relinking (IPR)
- Schuetz et al. (2022): Dual Annealing
- Mangussi et al. (2023): SA, ILS, and VNS
- Chaves et al. (2024): GRASP
- Chaves et al. (2025): RKO Solver

## Algorithm
The general structure of the RKO involves the following steps:

1. **Initialization**: Generate a population of random-key encoded solutions.
2. **Decoding**: Convert random-key vectors into feasible solutions using a problem-specific decoding function.
3. **Evaluation**: Assess the quality of each solution based on a predefined objective function.
4. **Search Process**:
   - Components:
      - Initial Solutions: Generate a diverse set of random-key vectors to explore the solution space.
      - Pool of Elite Solutions: Maintain a set of high-quality solutions for refinement and recombination.
      - Shaking: Introduce perturbations to escape local optima and enhance exploration.
      - Blending: Combine solutions to create new promising candidates.
      - Local Search: Apply improvement procedures to refine candidate solutions.
      - Metaheuristics: Incorporate additional optimization techniques (e.g., simulated annealing, VNS, ILS, ...) to guide the search process.
5. **Termination**: Repeat the process until a stopping criterion is met (e.g., time limit, maximum iterations or convergence threshold).

## Applications
RKO has been applied to a variety of combinatorial optimization problems, including:
- α-Neighborhood p-Median Problem (α-NpMP)
- α-Neighborhood p-Center Problem (α-NpCP)
- Node Capacitated Graph Partitioning Problem (NCGPP)
- Tree Hub Location Problem (THLP)
- Balanced Edge Partition Problem (BEPP)
- Operate Room Scheduling Problem (ORSP)
- Job Sequencing and tool Switching Problem (SSP)
- HEV Traveling Salesman Problem with Time Windows (HEVTSPTW)
- Steiner Triple Covering Problem (STCP)
- Travelling Thief Problem (TTP)
- One-Dimensional Multi-Period Cutting Stock Problem (MPCSP)
- Capacitated Vehicle Routing Problem (CVRP)
- Portfolio Optimization

## Advantages
- **Scalability**: Efficiently handles large-scale optimization problems.
- **Robustness**: Adapts to different problem structures with minimal customization.
- **Flexibility**: Can be integrated with other heuristic and exact methods for hybrid approaches.

## What's new 
- Continuous RKO algorithm for discrete optimization
- Multi-thread software framework with several RKOs that collaborate by way of solution sharing in a pool and through local search
- Most of the framework is problem-independent.  Only a problem-dependent decoder needs to be implemented by the user
- New adaptation of Nelder-Mead search for optimization in real n-dim unit hypercube
- Other simple local search procedures operate in continuous space and are integrated with the Nelder-Mead search in a Randomized Variable Neighborhood Descent (RVND) procedure
- The software is on GitHub

## References
- Andrade, C.E., Byers, S.D., Gopalakrishnan, V., Halepovic, E., Poole, D.J., Tran, L.K., Volinsky, C.T., 2019. Scheduling software updates for connected cars with limited availability. Applied Soft Computing 82, 105575. doi:10.1016/j.asoc.2019.105575.
- Andrade, C.E., Toso, R.F., Gon¸calves, J.F., Resende, M.G., 2021. The multi-parent biased random-key genetic algorithm with implicit path-relinking and its real-world applications. European Journal of Operational Research 289, 17–30. doi:10.1016/j.ejor.2019.11.037.
- Bean, J.C., 1994. Genetic algorithms and random keys for sequencing and optimization. ORSA Journal on Computing 6, 154–160. URL: 10.1007/s10729-008-9080-9.
- Chaves, A.A., Resende, M.G.C., Silva, R.M.A., 2024. A random-key grasp for combinatorial optimization. Journal of Nonlinear & Variational Analysis 8. doi:10.23952/jnva.8.2024.6.03.
- Garcia-Santiago, C., Del Ser, J., Upton, C., Quilligan, F., Gil-Lopez, S., Salcedo-Sanz, S., 2015. A random-key encoded harmony search approach for energy-efficient production scheduling with shared resources. Engineering Optimization 47, 1481–1496. doi:10.1080/0305215X.2014.
- Gonçalves, J.F., De Almeida, J.R., 2002. A hybrid genetic algorithm for assembly line balancing. Journal of Heuristics 8, 629–642. doi:10.1023/A:1020377910258.
- Gonçalves, J.F., Resende, M.G.C., 2011. Biased random-key genetic algorithms for combinatorial optimization. Journal of Heuristics 17, 487–525. doi:10.1007/s10732-010-9143-1.
- Lin, T.L., Horng, S.J., Kao, T.W., Chen, Y.H., Run, R.S., Chen, R.J., Lai, J.L., Kuo, I.H., 2010. An efficient job-shop scheduling algorithm based on particle swarm optimization. Expert
Systems with Applications 37, 2629–2636. doi:10.1016/j.eswa.2009.08.015.
- Londe, M.A., Pessoa, L.S., Andrade, C.E., Resende, M.G., 2025. Biased random-key genetic algorithms: A review. European Journal of Operational Research 321, 1–22. doi:10.1016/j.ejor.2024.03.030.
- Mangussi, A.D., Pola, H., Macedo, H.G., ao, L.A.J., Proen¸ca, M.P.T., Gianfelice, P.R.L., Salezze, B.V., Chaves, A.A., 2023. Meta-heurísticas via chaves aleatórias aplicadas ao problema de localização de hubs em árvore, in: Anais do Simpósio Brasileiro de Pesquisa Operacional, Galoá, São José dos Campos. p. 25. doi:10.59254/sbpo-2023-174934.
- Noronha, T.F., Ribeiro, C.C., 2024. Biased random-key genetic algorithms: A tutorial with applications. ACM International Conference Proceeding Series , 110 – 115doi:10.1145/3665065.
- Pessoa, L.S., Andrade, C.E., 2018. Heuristics for a flowshop scheduling problem with stepwise job objective function. European Journal of Operational Research 266, 950–962. doi:10.1016/j.ejor.2017.10.045.
- Schuetz, M.J., Brubaker, J.K., Montagu, H., van Dijk, Y., Klepsch, J., Ross, P., Luckow, A., Resende, M.G., Katzgraber, H.G., 2022. Optimization of robot-trajectory planning with nature-inspired and hybrid quantum algorithms. Phys. Rev. Appl. 18, 054045. doi:10.1103/ PhysRevApplied.18.054045.

## External Links
- github.com/RKO-solver
- Antonio A. Chaves and Mauricio G. C. Resende and Martin J. A. Schuetz and J. Kyle Brubaker and Helmut G. Katzgraber and Edilson F. de Arruda and Ricardo M. A. Silva. A Random-Key Optimizer for Combinatorial Optimization}, 2025, arXiv, https://arxiv.org/abs/2411.04293.
