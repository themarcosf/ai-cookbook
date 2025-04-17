import SEAL from "node-seal";

(async () => {
  const seal = await SEAL();

  const count = seal.CoeffModulus.MaxBitCount(4096, seal.SecurityLevel.tc256);
  console.log("coeffModulus::maxBitCount: ", count);

  const bfvDefault = seal.CoeffModulus.BFVDefault(4096, seal.SecurityLevel.tc256);
  console.log("coeffModulus::bfvDefault: ", bfvDefault.toArray());

  const created = seal.CoeffModulus.Create(4096, Int32Array.from([36, 36, 37]));
  console.log("coeffModulus::create: ", created.toArray());
})();

// Ref: https://github.com/s0l0ist/node-seal/blob/9618aca13e745ebb2a9c7d3a6b18d78e68d4aab9/src/__tests__/coeff-modulus.test.ts