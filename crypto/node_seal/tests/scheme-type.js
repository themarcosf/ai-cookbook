import SEAL from "node-seal";

(async () => {
  const seal = await SEAL();

  const _none = seal.SchemeType.none;
  const _bfv = seal.SchemeType.bfv;
  const _bgv = seal.SchemeType.bgv;
  const _ckks = seal.SchemeType.ckks;

  console.log("schemeType::none: ", _none.constructor.name);
  console.log("schemeType::bfv: ", _bfv.constructor.name);
  console.log("schemeType::bgv: ", _bgv.constructor.name);
  console.log("schemeType::ckks: ", _ckks.constructor.name);
})();

// Ref: https://github.com/s0l0ist/node-seal/blob/main/src/__tests__/scheme-type.test.ts
