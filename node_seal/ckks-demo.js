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
  console.log("\n");

  const plain = Float64Array.from({ length: ckksEncoder.slotCount }).map((_, i) => i)
  console.log(`plain: ${plain.slice(0, 5)}...`);

  const encoded = ckksEncoder.encode(plain, Math.pow(2, 20));
  console.log(`encoded: ${encoded.save().slice(0, 50)}...`);

  const decoded = ckksEncoder.decode(encoded);
  console.log(`decoded: ${decoded.slice(0, 10)}...`);
})();

// Ref: https://github.com/s0l0ist/node-seal/tree/9618aca13e745ebb2a9c7d3a6b18d78e68d4aab9