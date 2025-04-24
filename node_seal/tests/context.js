import SEAL from "node-seal";

(async () => {
  const seal = await SEAL();

  const bfvEncParms = seal.EncryptionParameters(seal.SchemeType.bfv)
  bfvEncParms.setPolyModulusDegree(4096)
  bfvEncParms.setCoeffModulus(seal.CoeffModulus.BFVDefault(4096, seal.SecurityLevel.tc128))
  bfvEncParms.setPlainModulus(seal.PlainModulus.Batching(4096, 20))

  const bfvContext = seal.Context(bfvEncParms)
  console.log("bfvContext::parametersSet: ", bfvContext.parametersSet())
  console.log("bfvContext::usingKeyswitching: ", bfvContext.usingKeyswitching)
  console.log("bfvContext::toHuman: ", bfvContext.toHuman())
  console.log("bfvContext::keyContextData::totalCoeffModulusBitCount: ", bfvContext.keyContextData.totalCoeffModulusBitCount)
  console.log("bfvContext::keysParmsId::value: ", bfvContext.keyParmsId.values)
  console.log("bfvContext::data::firstContextData::totalCoeffModulusBitCount: ", bfvContext.firstContextData.totalCoeffModulusBitCount)
  console.log("bfvContext::data::firstParmsId::totalCoeffModulusBitCount: ", bfvContext.firstParmsId.values)
  console.log("bfvContext::data::lastContextData::totalCoeffModulusBitCount: ", bfvContext.lastContextData.totalCoeffModulusBitCount)
  console.log("bfvContext::data::lastParmsId::totalCoeffModulusBitCount: ", bfvContext.lastParmsId.values)
})();

// Ref: https://github.com/s0l0ist/node-seal/blob/9618aca13e745ebb2a9c7d3a6b18d78e68d4aab9/src/__tests__/context.test.ts