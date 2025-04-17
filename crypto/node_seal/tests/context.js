import SEAL from "node-seal";

(async () => {
  const seal = await SEAL();

  const bfvEncParms = seal.EncryptionParameters(seal.SchemeType.bfv)
  bfvEncParms.setPolyModulusDegree(4096)
  bfvEncParms.setCoeffModulus(seal.CoeffModulus.BFVDefault(4096, seal.SecurityLevel.tc128))
  bfvEncParms.setPlainModulus(seal.PlainModulus.Batching(4096, 20))

  const ckksEncParms = seal.EncryptionParameters(seal.SchemeType.ckks)
  ckksEncParms.setPolyModulusDegree(4096)
  ckksEncParms.setCoeffModulus(seal.CoeffModulus.Create(4096, Int32Array.from([46, 16, 46])))

  const bfvContext = seal.Context(bfvEncParms)
  const parmsId = bfvContext.firstParmsId
  console.log("bfvContext::toHuman: ", bfvContext.toHuman())
  console.log("bfvContext::paramsId: ", bfvContext.getContextData(parmsId).totalCoeffModulusBitCount)
  console.log("bfvContext::keys: ", bfvContext.keyContextData.totalCoeffModulusBitCount)
  console.log("bfvContext::data::firstContext: ", bfvContext.firstContextData.totalCoeffModulusBitCount)
  console.log("bfvContext::data::lastContext: ", bfvContext.lastContextData.totalCoeffModulusBitCount)
})();

// Ref: https://github.com/s0l0ist/node-seal/blob/9618aca13e745ebb2a9c7d3a6b18d78e68d4aab9/src/__tests__/context.test.ts