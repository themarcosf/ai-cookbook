import SEAL from "node-seal";

(async () => {
  const seal = await SEAL();

  const encParams = seal.EncryptionParameters();
  console.log("encParams::instance: ", encParams.instance.constructor.name);
  console.log("encParams::scheme: ", encParams.scheme.constructor.name);
  console.log("encParams::parmsId: ", encParams.parmsId.values);
  console.log("\n");
  
  const bfvEncParams = seal.EncryptionParameters(seal.SchemeType.bfv);
  const bfvCoeffModulus = seal.CoeffModulus.BFVDefault(4096, seal.SecurityLevel.tc128);
  bfvEncParams.setPolyModulusDegree(4096);
  bfvEncParams.setCoeffModulus(bfvCoeffModulus);
  bfvEncParams.setPlainModulus(seal.Modulus(BigInt('786433')))
  console.log("bfvEncParams::polyModulusDegree: ", bfvEncParams.polyModulusDegree);
  console.log("bfvEncParams::coeffModulus: ", bfvEncParams.coeffModulus);
  console.log("bfvEncParams::plainModulus: ", bfvEncParams.plainModulus.value);
  console.log("\n");
  
  const bgvEncParams = seal.EncryptionParameters(seal.SchemeType.bgv);
  bgvEncParams.setPolyModulusDegree(4096);
  bgvEncParams.setCoeffModulus(bfvCoeffModulus);
  bgvEncParams.setPlainModulus(seal.Modulus(BigInt('786433')))
  console.log("bgvEncParams::polyModulusDegree: ", bgvEncParams.polyModulusDegree);
  console.log("bgvEncParams::coeffModulus: ", bgvEncParams.coeffModulus);
  console.log("bgvEncParams::plainModulus: ", bgvEncParams.plainModulus.value);
  console.log("\n");
  
  const ckksEncParams = seal.EncryptionParameters(seal.SchemeType.ckks);
  const ckksCoeffModulus = seal.CoeffModulus.Create(4096, Int32Array.from([46, 16, 46]));
  ckksEncParams.setPolyModulusDegree(4096);
  ckksEncParams.setCoeffModulus(ckksCoeffModulus);
  console.log("ckksEncParams::polyModulusDegree: ", ckksEncParams.polyModulusDegree);
  console.log("ckksEncParams::coeffModulus: ", ckksEncParams.coeffModulus);
  console.log("\n");
})();

// Ref: https://github.com/s0l0ist/node-seal/blob/main/src/__tests__/encryption-parameters.test.ts