# Wind farm layout optimization using quantum computing

The global transition towards renewable energy sources is crucial for mitigating climate change and ensuring a sustainable future. Wind energy, as a clean and abundant renewable resource, plays a pivotal role in this transition. However, maximizing the efficiency and sustainability of wind farms requires careful optimization of wind turbine placement, a complex problem known as wind farm layout optimization (WFLO).

WFLO involves determining the optimal positions of wind turbines within a wind farm to maximize energy production while minimizing adverse environmental impacts and economic costs. The challenge lies in balancing these objectives, as placing turbines too close together can lead to wake effects that reduce overall energy output while placing them too far apart can increase land usage and installation costs.

Classical optimization methods, such as genetic algorithms and particle swarm optimization, have been used for WFLO, but they often struggle with large-scale problems due to computational limitations. Quantum computing, with its ability to harness the power of quantum mechanics, offers a promising alternative for solving complex optimization problems like WFLO.

## Hybrid Quantum-Classical Optimization 

To address the WFLO challenge, we propose a hybrid quantum-classical optimization approach. This approach has the potential to significantly improve the efficiency and effectiveness of wind farm layout optimization, thereby contributing to the goals of Sustainable Development Goal 7 (SDG 7) to ensure access to affordable, reliable, sustainable, and modern energy for all. 

The proposed methodology assumes a wind farm as a grid due to which we can formulate the WFLO problem as a quadratic unconstrained binary optimization (QUBO) problem, which can be solved using QAOA or QAA. The QUBO formulation captures the objectives of maximizing energy production, and minimizing wake effects with proximity constraints taken into account [1]. More complex systems can also modelled for this problem which include surface roughness of the terrain, wake decay etc. depending on the timeframe.

To scale up to a larger grid, we use problem decomposition strategies which enable partitioning the QUBO into subQUBOs, solving them and combining the results. This is based on the qbsolv algorithm [2]. We solve the subQUBOs using Pasqal's neutral atom devices. The algorithm makes a set of calls to a subQUBO solver for global minimization and a call to classical search (Tabu [3]) for local minimization. 

### Small grid size

For an initial prototype, we start with smaller grid sizes which can fit on current Pasqal devices. This involves using the QAOA and/or QAA to solve the QUBO formulation \cite{pulser}. The QUBO problem is embedded onto atomic register followed by preparing the following ising Hamiltonian

$$ H_Q= \sum_{i=1}^N \frac{\hbar\Omega}{2} \sigma_i^x - \sum_{i=1}^N \frac{\hbar \delta}{2} \sigma_i^z+\sum_{j \lt i}\frac{C_6}{|\textbf{r}_i-\textbf{r}_j|^{6}} n_i $$

### Scaling grid size

For large grid sizes, which cannot fit on the current Pasqal devices, we use decomposition techniques as presented in [2]. The approach is to select subQUBOs of variables which contribute maximally to the problem energy.

## Sustainable Use-Case: Enhancing Wind Energy Efficiency and Reducing Environmental Impacts

The proposed hybrid quantum-classical optimization approach for WFLO directly aligns with the UN's SDG 7 – Affordable and Clean Energy. 

By optimizing wind farm layouts, we can increase energy production from renewable sources, reduce the cost of wind energy, and minimize the environmental impact of wind farms. Even a tiny amount of progress made in this direction can save millions of tonnes of carbon emissions.

## Quantum Project Relevance: Addressing a Real-World Challenge with Quantum Computing

The proposed project directly addresses a real-world challenge in the renewable energy sector, demonstrating the potential of quantum computing to solve complex optimization problems that have significant societal and environmental impacts. 

The hybrid design takes advantage of both classical search and the QAOA/QAA. While the quantum computer is highly effective at exploring various parts of the state space it can get trapped in local minima. On the other hand, the tabu search can locate an exact minimum inside a neighbourhood very rapidly, but it occasionally has trouble leaving it. The method can be thought of as a large-neighborhood local search, where each iteration produces tabu improvements [2].

## Directory Structure

`pasqal_hybrid` is the directory containing the code for QAOA and QAA samplers which can be called from the `main.py` file.  `main.py` formulates the QUBO problem and can be executed using the following command

```
python main.py --m 20 --n_wind 36 --axis 10
```
where `m` is the number of turbines to place on the wind farm, `n_wind` is the wind regime (currently supports only 1 and 36 wind regimes) and `axis` is the grid size i.e. the size of the grid will be `axis`$\times$`axis`. 

### Samplers

`pasqal_hybrid` supports currently two samplers: `PasqalQAOAProblemSampler` and `PasqalQAAProblemSampler` along with their classes for solving subproblems. These can be used in the `main.py` (line 316). `pasqal_utils` has functions to embed QUBO on neutral atoms using classical optimization (Nelder-mead).

**Note:** See the document for results and analysis.

# References

1. A. Senderovich, J. Zhang, E. Cohen and J. C. Beck, "Exploiting Hardware and Software Advances for Quadratic Models of Wind Farm Layout Optimization," in IEEE Access, vol. 10, pp. 78044-78055, 2022, doi: 10.1109/ACCESS.2022.3193143.
2. Booth, Michael, Steven P. Reinhardt and Aidan Roy. “Partitioning Optimization Problems for Hybrid Classical/Quantum Execution TECHNICAL REPORT.” (2017).
3. Glover, F., Laguna, M. (1998). Tabu Search. In: Du, DZ., Pardalos, P.M. (eds) Handbook of Combinatorial Optimization. Springer, Boston, MA. https://doi.org/10.1007/978-1-4613-0303-9_33
