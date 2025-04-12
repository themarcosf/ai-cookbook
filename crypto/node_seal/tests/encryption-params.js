import SEAL from "node-seal";

(async () => {
  const seal = await SEAL();

  const encParams = seal.EncryptionParameters();

  console.log("Instance: ", encParams.instance.constructor.name);
  console.log("Scheme: ", encParams.scheme.constructor.name);
  console.log("polyModulusDegree: ", encParams.polyModulusDegree);
  console.log("coeffModulus: ", encParams.coeffModulus);
  console.log("plainModulus: ", encParams.plainModulus);

  // expect(encParms).toHaveProperty('unsafeInject')
  // expect(encParms).toHaveProperty('delete')
  // expect(encParms).toHaveProperty('setPolyModulusDegree')
  // expect(encParms).toHaveProperty('setCoeffModulus')
  // expect(encParms).toHaveProperty('setPlainModulus')
  // expect(encParms).toHaveProperty('parmsId')
  // expect(encParms).toHaveProperty('save')
  // expect(encParms).toHaveProperty('saveArray')
  // expect(encParms).toHaveProperty('load')
  // expect(encParms).toHaveProperty('loadArray')
})();

// Ref: https://github.com/s0l0ist/node-seal/blob/main/src/__tests__/encryption-parameters.test.ts
