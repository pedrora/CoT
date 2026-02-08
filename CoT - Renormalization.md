# **Appendix — Phase-Coherent Renormalization Across Semantic Scales**

## A.1 Motivation

The CoT / ToAE framework operates across multiple descriptive scales, ranging from low-level symbolic representations $characters, tokens$ to high-level constructs $concepts, propositions, narratives$.\
To avoid arbitrary scale jumps or metaphorical abstraction, transitions between these scales must preserve **coherence invariants**.

This Appendix formalizes those transitions using a **renormalization framework**, analogous in spirit $but not ontology$ to Wilsonian renormalization in physics, adapted here to **informational and semantic fields**.

***

## A.2 Layered State Spaces

Let there exist a hierarchy of complex state spaces:

$$\
\mathcal{H}\_0 \rightarrow \mathcal{H}\_1 \rightarrow \mathcal{H}\_2 \rightarrow \dots \rightarrow \mathcal{H}\_n\
$$

where:

* $\mathcal{H}\_0$ represents the finest resolution (e.g. character-level states),

* higher $\mathcal{H}\_k$ represent progressively coarser semantic resolutions $tokens, concepts, propositions, narratives$.

Each $\mathcal{H}\_k$ is endowed with:

* complex amplitudes $\psi \in \mathcal{H}\_k$,

* explicit phase information $\arg(\psi)$,

* a coherence functional $\kappa_k(\psi)$.

No assumption is made that these spaces are linguistic in nature; the same structure applies to physical, cognitive, or symbolic domains.

***

## A.3 Renormalization Operator

Define the **phase-coherent renormalization operator**:

$$\
\mathcal{R}\_{k \rightarrow k+1} : \mathcal{H}_k \rightarrow \mathcal{H}_{k+1}\
$$

subject to the following constraints.

***

### A.3.1 Phase Covariance

$$\
\mathcal{R}_{k \rightarrow k+1}(e^{i\theta}\psi)
=\
e^{i\theta}\mathcal{R}_{k \rightarrow k+1}(\psi)\
$$

Global phase alignment is preserved under scale transition.\
This ensures that truth evaluation based on phase alignment remains well-defined across resolutions.

***

### A.3.2 Coherence Monotonicity

$$\
\kappa_{k+1}(\mathcal{R}_{k \rightarrow k+1}(\psi))\
\ge\
\kappa_k(\psi) - \epsilon_k\
\quad\text{with finite } \epsilon_k\
$$

Renormalization may lose information, but **cannot create coherence ex nihilo**.\
Structures that are incoherent at lower scales cannot become coherent solely through abstraction.

***

### A.3.3 Dimensional Compression

$$\
\dim(\mathcal{H}_{k+1}) < \dim(\mathcal{H}_k)\
$$

Renormalization reduces descriptive degrees of freedom while preserving phase-stable invariants.\
This enforces genuine abstraction rather than duplication.

***

## A.4 Explicit Construction

Partition $\mathcal{H}\_k$ into disjoint blocks ${\mathcal{B}\_j}$, each representing a coherent cluster (e.g. characters forming a token, tokens forming a concept).

The renormalized state is defined as:

$$\
(\mathcal{R}_{k \rightarrow k+1}\psi)_j
=\
\frac{1}{Z_j}\
\sum_{x \in \mathcal{B}_j}\
w_x , \psi_x\
$$

where:

* $w_x$ are context-dependent weights,

* $Z_j$ is a normalization factor.

This construction preserves phase information while allowing destructive interference to cancel incoherent components.

***

## A.5 Phase Stability Criterion

A block $\mathcal{B}_j$ is considered **semantically stable** iff:

$$\
\left|\
\frac{1}{|\mathcal{B}_j|}\
\sum_{x \in \mathcal{B}_j}\
e^{i \arg(\psi_x)}\
\right|\
\ge \tau\
$$

where $\tau$ is a minimum coherence threshold.

Failure of this condition results in:

* amplitude cancellation,

* loss of coherence,

* increased bullshit / instability scores in CoT evaluation.

***

## A.6 Fixed Points and Truth Persistence

A structure $\psi^\*$ is said to be **realized** within the CoT framework iff it is a fixed point under recursive renormalization:

$$\
\psi^\*
=\
\mathcal{R}_{k \rightarrow k+1}\
\circ\
\mathcal{R}_{k-1 \rightarrow k}\
\circ \dots\
(\psi^\*)\
$$

Truth, in this sense, is not correspondence to an external label, but **stability under recursive, phase-coherent scale transitions**.

This criterion applies uniformly to:

* physical constants,

* cognitive identities,

* semantic propositions,

* narrative structures.

***

## A.7 Interpretation

Renormalization in CoT is neither metaphorical nor purely statistical.\
It is a **structural constraint** enforcing that only phase-coherent patterns survive abstraction.

Bullshit is thus formally characterized as **destructive interference across scales**, not as mere improbability or ignorance.

***

### Appendix Summary

> _Renormalization in CoT is defined as a phase-covariant, coherence-preserving projection between state spaces of decreasing resolution. Truth corresponds to structures that remain fixed points under this transformation._
