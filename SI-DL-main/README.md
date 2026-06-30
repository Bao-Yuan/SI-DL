# SI-DL: Dimensionless Learning with Conditional Mean-Variance Sobol Index

This folder mirrors the `IT_PI-main` example layout, but replaces mutual
information with the conditional mean-variance Sobol index

\[
S(U)=\frac{\mathrm{Var}(\mathbb{E}[Y\mid U])}{\mathrm{Var}(Y)}
\]

estimated through the covariance identity

\[
\widehat S_{\mathrm{cov}}(U)=
\frac{\mathrm{Cov}(Y,\widehat m(U))}{\mathrm{Var}(Y)}.
\]

The conditional mean \(\widehat m(U)\) is estimated with Gaussian
distance-weighted kNN, using a default bandwidth scale of 0.5 times the
distance to the \(k\)-th neighbor. When `k=None`, the neighborhood size is
chosen automatically by K-fold cross-validation. The candidate set is formed
by uniformly sampling 10 integer values in
\([\max(5, 0.2n^{4/(d+4)}), n^{4/(d+4)}]\), where \(n\) is the sample size and \(d\) is
the candidate input dimension. For each candidate \(k\), the code computes the
K-fold kNN conditional-mean prediction error and uses the \(k\) with the
smallest error. Boundary bias is handled with
mirrored samples in the kNN fit by default. The mirror rule is enabled for
one- and two-dimensional candidates by default; in 2D it fits against \(9n\)
samples, covering the original points, four edge reflections, and four corner
reflections. Higher dimensions require `mirror_max_dim=None` because the
augmented fit size is \(n3^d\). Edge trimming remains available as an optional
scoring choice. For dimensional physical examples, candidate inputs
are restricted to Buckingham-\(\pi\) groups. The exponent vectors are generated
from the null space of `D_in`, following the same convention as `IT_PI.py`.

## Structure

- `SI_DL.py`: core estimator, Buckingham-\(\pi\) utilities, and differential
  evolution search.
- `Examples/Mathematical/Mathematical_examples.ipynb`: synthetic examples
  without dimensional analysis.
- `Examples/Colebrook/Colebrook.ipynb`: Colebrook-White equation in
  dimensionless form.
- `Examples/Benard/Benard_convection.ipynb`: Rayleigh-Benard convection in
  dimensionless form.

## How `D_in` Is Built

Let the dimensional inputs be \(q_1,\ldots,q_p\). Each column of `D_in` is the
base-dimension exponent vector of one input variable, in the same order as the
columns of `X`. A power-law group

\[
\Pi = \prod_j q_j^{\omega_j}
\]

is dimensionless exactly when

\[
D_{\mathrm{in}}\omega = 0.
\]

`SI_DL.calc_basis(D_in, num_basis)` uses the IT-PI convention to construct basis
vectors for this null space. Differential evolution then searches linear
combinations of these basis vectors.
