# A semi-infinite hydraulic fracture propagating in a poroelastic medium

This repository accompanies the paper **"A semi-infinite hydraulic fracture propagating in a poroelastic medium"**.

It contains a numerical solver for a steadily propagating semi-infinite hydraulic fracture in a poroelastic medium, together with notebooks for reproducing the fracture profiles, parameter distributions, relative difference maps and poroelastic coupling regime maps presented in the paper.

## Repository contents

```text
poroelastic-hydraulic-fracture-tip/
├── data/
│   ├── precomputed_integrals/
│   └── precomputed_results/
├── exports/
│   ├── near_tip_results.html
│   └── integral_precomputation.html
├── notebooks/
│   ├── near_tip_results.ipynb
│   └── integral_precomputation.ipynb
├── src/
│   ├── __init__.py
│   ├── asymptotes.py
│   ├── collocation_utils.py
│   ├── GC_calculus.py
│   ├── integration_utils.py
│   ├── plotting.py
│   ├── poroelastic_kernels.py
│   ├── precomputed_integrals.py
│   └── solver.py
├── LICENSE
├── README.md
└── requirements.txt
```

## Description

The code combines the poroelastic boundary integral formulation with lubrication flow inside the fracture and the fracture propagation condition. It includes:

- dimensionless poroelastic kernel functions;
- Gauss–Chebyshev quadrature and collocation routines;
- numerical integration utilities and precomputed integral matrices;
- the fully coupled near-tip solver and reduced formulations with two-dimensional pressure-dependent fluid exchange;
- near-tip and far-field asymptotic profiles;
- comparisons of fracture opening, net pressure, and fluid displacement;
- relative difference maps and maps identifying the dominant poroelastic coupling mechanisms.

The numerical problem is solved in diffusion scaling. The solver returns the fracture profiles and stress and pressure contributions in viscosity–toughness $mk-$scaling.

## Reproducing the results

The main notebook is:

```text
notebooks/near_tip_results.ipynb
```

It covers:

1. representative governing parameters for sandstones, granites, and shales (Section 5.1), together with the distributions of the characteristic length $\ell_{mk}$ (Figure S3);
2. the effects of dimensionless permeability $\varrho$, effective confining stress $\Sigma_0'$, and fracture propagation velocity $V$ (Section 5.2);
3. the effects of the poroelastic parameters $\beta$ and $\eta$ (Appendix D);
4. relative difference and coupling regime maps for fully coupled and reduced models (Section 5.3 and Appendix E);
5. fracture profiles and individual stress and pressure contributions in characteristic coupling regimes (Section 5.4).

The default settings reproduce selected transects and representative points. Alternative cases from the paper are provided as commented parameter settings. To reproduce another case, change these settings and rerun the calculation and plotting cells in the corresponding section. For the relative difference maps, select the reduced model using `model_name`.

The demonstration of the $\ell_{mk}$ distributions uses fewer parameter combinations than the precomputed lookup. The subsequent calculations use the precomputed mean $\ell_{mk}$ values obtained from denser sampling of the same parameter ranges.

## Precomputed data

The notebooks use the following folders:

```text
data/precomputed_integrals/
data/precomputed_results/
```

The first contains the near-tip integral vectors and interval integral matrices for the $S_e$, $P_e$, $P_s$, and Cauchy operators. The $S_e$ arrays are stored for $\beta = 0.25$ and reconstructed for the requested value of $\beta$. The $S_s$ operator uses the same discrete coupling matrix as $P_e$.

The second contains:

| File | Purpose |
| --- | --- |
| `l_mk_mean_min_max.npy` | Lookup of $\ell_{mk}$ statistics across the parameter space |
| `fully_coupled.npy` | Fully coupled fracture profiles |
| `uncoupled.npy` | Uncoupled fracture profiles |
| `one_way_coupled.npy` | One-way coupled fracture profiles |
| `no_cross_couplings.npy` | Profiles for the auxiliary model without cross-coupling terms |

The parameter-space solution arrays contain distance $\xi$, opening $\Omega$, and net pressure $\Pi$, with axes ordered as $(\Sigma_0', \varrho, \xi, \mathrm{field})$.

The default discretization uses 300 primary nodes and the mapping exponent `p = 3`. The quadrature object and integral arrays must correspond to the same discretization. 

The full parameter-space calculation is included as a commented example in the main notebook. The distributed results are loaded for constructing the relative difference and regime maps. When forming the reduced models, the dimensionless storage coefficient $\mathcal{S}$ is retained at the value used for the fully coupled model.

## Integral matrix precomputation

The notebook

```text
notebooks/integral_precomputation.ipynb
```

demonstrates the evaluation of the integral matrices used by the solver. It calculates the arrays in memory and does not overwrite the files in `data/precomputed_integrals`.

The numerical integration routines are implemented in `src/integration_utils.py`.

## Static exported files

The `exports/` folder contains static HTML versions of the notebooks, including their saved outputs, for convenient viewing without running Python or Jupyter. Download an HTML file and open it in a web browser.
## Installation

Clone the repository:

```bash
git clone https://github.com/evgenii-kanin/poroelastic-hydraulic-fracture-tip.git
cd poroelastic-hydraulic-fracture-tip
```

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

The dependencies are NumPy, SciPy, Matplotlib, mpmath, tqdm, and Jupyter. Matplotlib 3.8 or newer is required by the contour plotting routines.

## Running the notebooks

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Open `notebooks/near_tip_results.ipynb` or `notebooks/integral_precomputation.ipynb` and run the cells in order from a fresh kernel. The main notebook requires the precomputed data listed above.

## Related repository

The code accompanying the paper, *"Steadily moving semi-infinite fracture in plane poroelasticity"*, devoted to the derivation of boundary integral equations, is available in [semi-infinite-fracture-poroelasticity](https://github.com/evgenii-kanin/semi-infinite-fracture-poroelasticity).

## Citation

If you use this repository, please cite the accompanying paper:

> A semi-infinite hydraulic fracture propagating in a poroelastic medium.

## License

The code is released under the BSD 3-Clause License. See [LICENSE](LICENSE) for the license terms.
