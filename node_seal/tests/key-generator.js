import SEAL from "node-seal";

(async () => {
  const seal = await SEAL();

  const schemeType = seal.SchemeType.bfv
  const securityLevel = seal.SecurityLevel.tc128
  const polyModulusDegree = 4096
  const bitSize = 20

  const bfvCoeffModulus = seal.CoeffModulus.BFVDefault(polyModulusDegree)
  const bfvPlainModulus = seal.PlainModulus.Batching(polyModulusDegree, bitSize)
  const bfvEncParms = seal.EncryptionParameters(schemeType)
  bfvEncParms.setPolyModulusDegree(polyModulusDegree)
  bfvEncParms.setCoeffModulus(bfvCoeffModulus)
  bfvEncParms.setPlainModulus(bfvPlainModulus)
  const bfvContext = seal.Context(bfvEncParms, true, securityLevel)

  const keyGenerator = seal.KeyGenerator(bfvContext)

  console.log(`keyGenerator::secretKey: ${keyGenerator.secretKey().save().slice(0, 50)}...`)

  const pubKey = keyGenerator.createPublicKey()
  console.log(`keyGenerator::publicKey: ${pubKey.save().slice(0, 50)}...`)

  const relinKey = keyGenerator.createRelinKeys()
  console.log(`keyGenerator::relinearizationKey: ${relinKey.save().slice(0, 50)}...`)

  const galoisKeys = keyGenerator.createGaloisKeys()
  console.log(`keyGenerator::galoisKeys::general: ${galoisKeys.save().slice(0, 50)}...`)

  const galoisKeys2 = keyGenerator.createGaloisKeys(
    Int32Array.from([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
  )
  console.log(`keyGenerator::galoisKeys::specific: ${galoisKeys2.save().slice(0, 50)}...`)
})();

// Ref: https://github.com/s0l0ist/node-seal/blob/9618aca13e745ebb2a9c7d3a6b18d78e68d4aab9/src/__tests__/galois-keys.test.ts