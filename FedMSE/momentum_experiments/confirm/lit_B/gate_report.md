| id | title match | total | passed | failed | sha256 |
|---|---|---|---|---|---|
| 1204.3982 | True | 3 | 0 | 3 | a29825d309efb17d |
| 1708.03888 | True | 5 | 4 | 1 | 8012e213ce51d5a1 |
| 1909.06335 | True | 7 | 7 | 0 | c1e068bbf4c34b1f |
| 2003.00295 | True | 5 | 5 | 0 | f3a0fd541ea39d3b |
| 2011.13584 | True | 5 | 5 | 0 | 7b0f85d6aaa18c33 |
| 2301.09604 | True | 4 | 2 | 2 | 5d5d6dce49157a75 |
| 2312.12670 | True | 4 | 4 | 0 | 0f6f38760b8ce45b |
| 2409.03137 | True | 5 | 5 | 0 | f3502dffb8758859 |
| 2505.11126 | True | 4 | 3 | 1 | eb03d4aa00e47e85 |
| 2510.04203 | True | 4 | 4 | 0 | d910f944f51e3e93 |
| 2511.22080 | True | 4 | 1 | 3 | 84d36f29b3b5fb60 |

FAILED 1204.3982: Function scheme: we restart whenever f(x^k) > f(x^{k−1}).

FAILED 1204.3982: For a smooth convex function the above scheme converges for any t_k ≤ 1/L; setting t_k = 1/L and q = 0 obtains a guaranteed convergence rate of

FAILED 1204.3982: Gradient scheme: we restart whenever ∇f(y^{k−1})^T (x^k − x^{k−1}) > 0.

FAILED 1708.03888: We use local LR λᵢ for each layer l:

FAILED 2301.09604: Update global model with w(t+1) ← w(t) − η(t)g ∆¯(t)

FAILED 2301.09604: For FedExP we optimize for  and ηl by grid search.

FAILED 2505.11126: Global Update (FedAvg): wt+1 = wt + ηg∆¯ t, where ηg is the global learning rate.

FAILED 2511.22080: $v_{b+1,k}^{r}=\alpha_{r}\,g_{b,k}^{r}+(1-\alpha_{r})\,\Delta_{r}^{k}$

FAILED 2511.22080: The main experiments are conducted with 100 clients, 10% participation per round, a batch size of 50, a local learning rate $\eta_{l}=0.1$, a global learning rate $\eta_{g}=1$, and five local epochs, running for 500 communication rounds.

FAILED 2511.22080: For efficiency, we compute the similarity using $\operatorname{sim}(\Delta_{r},\Delta_{r}^{k})$, which we empirically find to be a good proxy for the gradient–momentum similarity without incurring extra backpropagation (see Appendix B).
