URL: https://crypto.stanford.edu/craig/craig-thesis.pdf
Title: A fully homomorphic encryption scheme

> 1.1

- neural networks as 'ideal lattices' due to their ability to parallelize computations -- ie, neural networks perform ADD and MULT operations during backpropagation, which are the basis of homomorphic encryption

> 1.3

- THEOREM: "If E is bootstrappable, then, for any integer d, one can construct a scheme E(d) that can evaluate any circuit (consisting of NAND gates) of depth d. The decryption circuit for E(d) is the same as for E, and the complexity of encryption is also the same. E(d)’s public key size is O(d) times that of E’s. The complexity of EvaluateE(d) is polynomial in the security parameter and linear in the circuit size. If E is semantically secure against chosen plaintext attacks, then so is EvaluateE(d)"

  - this theorem formalizes how bootstrappable encryption schemes can be extended to support computations of arbitrary complexity
  - bootstrappability: a cryptographic scheme E is considered bootstrappable if it can evaluate its own decryption function on encrypted data. This means the scheme can “refresh” or “simplify” a ciphertext while still preserving the encrypted message.
  - circuit: a collection of logic gates (like NAND gates) used to compute a function. Circuits have a “depth,” which is the number of layers of gates.
  - complexity evaluation: the time complexity for evaluating a circuit is polynomial in the security parameter (a measure of cryptographic strength) and linear in the size of the circuit being evaluated.
  - semantic security: a cryptographic scheme is semantically secure if an attacker cannot infer any meaningful information about the plaintext given the ciphertext. If the original scheme E is semantically secure against chosen plaintext attacks (CPA-secure), then E(d) maintains this level of security. This ensures that the security of the scheme is not compromised as its capabilities are extended to evaluate deeper circuits.
  - E(d): a version of the encryption scheme E that can evaluate circuits up to a depth of d. The computational cost of encryption in E(d) is the same as that of E. This is important because it ensures scalability — constructing E(d) doesn’t make encryption more expensive.

- "One drawback of E(d) is that its public key is O(d) times that of E’s public key. Since E(d) has this unwanted dependence on d, we say that it is merely leveled fully homomorphic. Under certain assumptions, we can make the E(d) public key size be independent of d, in which case we say the derived scheme is fully homomorphic" --- our scheme is fully homomorphic

- "suppose that there is an “error” associated with each ciphertext [...] we could refresh a ciphertext if we could completely decrypt it, simply by generating an entirely new and fresh ciphertext that encrypts the same thing, but we want a way to refresh that does not require the secret key. This is the idea behind bootstrapping: we do decrypt the ciphertext, but homomorphically" --- in our case, by casting the input to an embedding space with arbitrary temperature

- idea of 'introducing new errors' -- in our case, the noise from the input embeddings

  - note that converting the encrypted input to a vector does not compromise semantic security

- jeweler analogy #2 -- approach to test further homomorphic encryption schemes

> 1.4

- ideal lattices as "typically dominated by matrix-vector multiplication"
