import SEAL from "node-seal";

(async () => {
  const seal = await SEAL();

  const _none = seal.SchemeType.none;
  const _bfv = seal.SchemeType.bfv;
  const _bgv = seal.SchemeType.bgv;
  const _ckks = seal.SchemeType.ckks;

  console.log("Scheme type is none: ", _none.constructor.name);
  console.log("Scheme type is BFV: ", _bfv.constructor.name);
  console.log("Scheme type is BGV: ", _bgv.constructor.name);
  console.log("Scheme type is CKKS: ", _ckks.constructor.name);
})();

// Ref: https://github.com/s0l0ist/node-seal/blob/main/src/__tests__/scheme-type.test.ts
