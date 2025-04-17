import SEAL from "node-seal";

(async () => {
  const seal = await SEAL();
  const polyModulusDegree = 4096;
  const bitSizes = Int32Array.from([46, 16, 46]);
  const securityLevel = seal.SecurityLevel.tc128;
  
  const coeffModulus = seal.CoeffModulus.Create(polyModulusDegree, bitSizes);
  const encParms = seal.EncryptionParameters(seal.SchemeType.ckks);
  encParms.setPolyModulusDegree(polyModulusDegree);
  encParms.setCoeffModulus(coeffModulus);
  const context = seal.Context(encParms, true, securityLevel);

  const ckksEncoder = seal.CKKSEncoder(context);
  console.log("ckksEncoder::slotCount: ", ckksEncoder.slotCount);

  const arr = Float64Array.from({ length: ckksEncoder.slotCount }).map((_, i) => i)

  const plain = seal.PlainText()
  ckksEncoder.encode(arr, Math.pow(2, 20), plain)
  console.log(`ckksEncoder::encode: ${plain.save().slice(0, 50)}...`);

  const plainDecoded = ckksEncoder.decode(plain);
  console.log(`ckksEncoder::decode: ${plainDecoded.slice(0, 10)}...`);

  const result = ckksEncoder.encode(arr, Math.pow(2, 20));
  console.log(`ckksEncoder::encode: ${result.save().slice(0, 50)}...`);

  const resultDecoded = ckksEncoder.decode(result);
  console.log(`ckksEncoder::decode: ${resultDecoded.slice(0, 10)}...`);
})();

// Ref: https://github.com/s0l0ist/node-seal/blob/9618aca13e745ebb2a9c7d3a6b18d78e68d4aab9/src/__tests__/ckks-encoder.test.ts