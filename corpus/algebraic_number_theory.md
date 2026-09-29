# Monograph on Algebraic Number Theory and Reciprocity Laws

## Chapter 1: Divisibility and Ideals in Number Fields
Let $K = \mathbb{Q}(\theta)$ be an algebraic number field and $\mathcal{O}_K$ its ring of integers.
A Dedekind domain is an integral domain in which every non-zero proper ideal factors uniquely into prime ideals.
The ideal class group $\text{Cl}(K)$ measures the failure of unique factorization in $\mathcal{O}_K$. Its cardinality $h_K = |\text{Cl}(K)|$ is the class number of $K$.

## Chapter 2: The Legendre and Jacobi Symbols
Let $p$ be an odd prime number. For any integer $a$, the Legendre symbol $\left(\frac{a}{p}\right)$ is defined as:
- $1$ if $a$ is a quadratic residue modulo $p$ and $a \not\equiv 0 \pmod p$.
- $-1$ if $a$ is a quadratic non-residue modulo $p$.
- $0$ if $a \equiv 0 \pmod p$.

Euler's Criterion states:
$$\left(\frac{a}{p}\right) \equiv a^{(p-1)/2} \pmod p.$$

Gauss's Lemma: Let $S = \{a, 2a, 3a, \dots, \frac{p-1}{2}a\}$. Reduce each element modulo $p$ into $(-p/2, p/2)$. If $\mu$ is the number of negative residues, then $\left(\frac{a}{p}\right) = (-1)^\mu$.

## Chapter 3: The Law of Quadratic Reciprocity
Theorem (Gauss, 1796): If $p$ and $q$ are distinct odd prime numbers, then:
$$\left(\frac{p}{q}\right)\left(\frac{q}{p}\right) = (-1)^{\frac{p-1}{2} \frac{q-1}{2}}.$$
Equivalently:
- If $p \equiv 1 \pmod 4$ or $q \equiv 1 \pmod 4$, then $\left(\frac{p}{q}\right) = \left(\frac{q}{p}\right)$.
- If $p \equiv q \equiv 3 \pmod 4$, then $\left(\frac{p}{q}\right) = -\left(\frac{q}{p}\right)$.

First Supplement:
$$\left(\frac{-1}{p}\right) = (-1)^{(p-1)/2} = \begin{cases} 1 & \text{if } p \equiv 1 \pmod 4, \\ -1 & \text{if } p \equiv 3 \pmod 4. \end{cases}$$

Second Supplement:
$$\left(\frac{2}{p}\right) = (-1)^{(p^2-1)/8} = \begin{cases} 1 & \text{if } p \equiv \pm 1 \pmod 8, \\ -1 & \text{if } p \equiv \pm 3 \pmod 8. \end{cases}$$

## Chapter 4: Cubic Reciprocity in Eisenstein Integers
Let $\omega = e^{2\pi i / 3} = \frac{-1 + \sqrt{-3}}{2}$. The ring of Eisenstein integers is $\mathbb{Z}[\omega]$.
The norm is $N(a + b\omega) = a^2 - ab + b^2$. A prime $\pi \in \mathbb{Z}[\omega]$ is primary if $\pi \equiv 2 \pmod 3$.
The cubic residue character $\left(\frac{\alpha}{\pi}\right)_3$ satisfies:
$$\left(\frac{\alpha}{\pi}\right)_3 \equiv \alpha^{(N(\pi)-1)/3} \pmod \pi.$$
For primary primes $\pi, \lambda$:
$$\left(\frac{\pi}{\lambda}\right)_3 = \left(\frac{\lambda}{\pi}\right)_3.$$

## Chapter 5: The Artin Reciprocity Law
Let $L/K$ be an abelian Galois extension of number fields.
The Artin map associates to each unramified prime ideal $\mathfrak{p}$ of $K$ the Frobenius element $\left(\frac{L/K}{\mathfrak{p}}\right) \in \text{Gal}(L/K)$.
The Artin Reciprocity Law establishes an isomorphism between the generalized ideal class group and the Galois group $\text{Gal}(L/K)$, unifying all classical reciprocity laws into class field theory.
