import SEAL from "node-seal";

(async () => {
  const seal = await SEAL();

  const polyModulusDegree = 4096
  const bitSize = 20

  const bfvCoeffModulus = seal.CoeffModulus.BFVDefault(polyModulusDegree)
  const bfvPlainModulus = seal.PlainModulus.Batching(polyModulusDegree, bitSize)
  const bfvEncParms = seal.EncryptionParameters(seal.SchemeType.bfv)
  bfvEncParms.setPolyModulusDegree(polyModulusDegree)
  bfvEncParms.setCoeffModulus(bfvCoeffModulus)
  bfvEncParms.setPlainModulus(bfvPlainModulus)
  const bfvContext = seal.Context(bfvEncParms)
  const bfvKeyGenerator = seal.KeyGenerator(bfvContext)

  const galoisKeys = seal.GaloisKeys();
  console.log(`galoisKey: ${galoisKeys.save().slice(0, 50)}...`);
  console.log("\n")

  console.log("galoisKeys::getIndex of 15: ", galoisKeys.getIndex(15));
  console.log("galoisKeys::getIndex of 3: ", galoisKeys.getIndex(3));
  console.log("galoisKeys::getIndex of 9: ", galoisKeys.getIndex(9));
  console.log("galoisKeys::getIndex of 11: ", galoisKeys.getIndex(11));
  console.log("\n")

  console.log("galoisKeys::hasKey of 3: ", galoisKeys.hasKey(3));

  const newKey = bfvKeyGenerator.createGaloisKeys()
  galoisKeys.move(newKey)
  console.log(`new galoisKey: ${galoisKeys.save().slice(0, 50)}...`);
})();

// Ref: https://github.com/s0l0ist/node-seal/blob/9618aca13e745ebb2a9c7d3a6b18d78e68d4aab9/src/__tests__/galois-keys.test.ts