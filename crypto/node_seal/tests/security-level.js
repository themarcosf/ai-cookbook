import SEAL from "node-seal";

(async () => {
  const seal = await SEAL();

  const _none = seal.SecurityLevel.none;
  const _tc128 = seal.SecurityLevel.tc128;
  const _tc192 = seal.SecurityLevel.tc192;
  const _tc256 = seal.SecurityLevel.tc256;

  console.log("Security level is none: ", _none.constructor.name);
  console.log("Security level is TC128: ", _tc128.constructor.name);
  console.log("Security level is TC192: ", _tc192.constructor.name);
  console.log("Security level is TC256: ", _tc256.constructor.name);
})();

// Ref: https://github.com/s0l0ist/node-seal/blob/main/src/__tests__/security-level.test.ts
