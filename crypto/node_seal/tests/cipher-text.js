import SEAL from "node-seal";

(async () => {
  const seal = await SEAL();

  const securityLevel = seal.SecurityLevel.tc128;
  const polyModulusDegree = 1024;
  const bitSizes = Int32Array.from([27]);
  const coeffModulus = seal.CoeffModulus.Create(polyModulusDegree, bitSizes);
  const sizeCapacity = 2;

  const bitSize = 20
  const plainModulus = seal.PlainModulus.Batching(polyModulusDegree, bitSize)
  const bfvEncParms = seal.EncryptionParameters(seal.SchemeType.bfv);
  bfvEncParms.setPolyModulusDegree(polyModulusDegree);
  bfvEncParms.setCoeffModulus(coeffModulus);
  bfvEncParms.setPlainModulus(plainModulus);
  const bfvContext = seal.Context(bfvEncParms, true, securityLevel);
  const bfvBatchEncoder = seal.BatchEncoder(bfvContext);
  const bfvKeyGenerator = seal.KeyGenerator(bfvContext);
  const bfvPublicKey = bfvKeyGenerator.createPublicKey();
  const bfvEncryptor = seal.Encryptor(bfvContext, bfvPublicKey);
  console.log("bfvBatchEncoder::slotCount: ", bfvBatchEncoder.slotCount);

  const ckksEncParms = seal.EncryptionParameters(seal.SchemeType.ckks);
  ckksEncParms.setPolyModulusDegree(polyModulusDegree);
  ckksEncParms.setCoeffModulus(coeffModulus);
  const ckksContext = seal.Context(ckksEncParms, true, securityLevel);
  const ckksEncoder = seal.CKKSEncoder(ckksContext);
  const ckksKeyGenerator = seal.KeyGenerator(ckksContext);
  const ckksPublicKey = ckksKeyGenerator.createPublicKey();
  const ckksEncryptor = seal.Encryptor(ckksContext, ckksPublicKey);
  console.log("ckksEncoder::slotCount: ", ckksEncoder.slotCount);
  console.log("\n");

  const bfvParmsId = bfvContext.firstParmsId
  const bfvCipher = seal.CipherText({context: bfvContext, parmsId: bfvParmsId, sizeCapacity})

  const bfvPlain = Int32Array.from({length: bfvBatchEncoder.slotCount}).fill(5);
  console.log(`bfvPlain: ${bfvPlain.slice(0, 5)}...`);

  const bfvEncoded = bfvBatchEncoder.encode(bfvPlain);
  console.log(`bfvEncoded: ${bfvEncoded.save().slice(0, 50)}...`);

  bfvEncryptor.encrypt(bfvEncoded, bfvCipher);
  console.log("bfvCipher::parmsId: ", bfvCipher.parmsId.values);
  console.log("bfvCipher: ", bfvCipher.save().slice(0, 50));
  console.log("bfvCipher::isTransparent: ", bfvCipher.isTransparent);
  console.log("bfvCipher::isNttForm: ", bfvCipher.isNttForm);
  console.log("\n");

  const ckksParmsId = ckksContext.firstParmsId
  const ckksCipher = seal.CipherText({context: ckksContext, parmsId: ckksParmsId, sizeCapacity})

  const ckksPlain = Float64Array.from({length: ckksEncoder.slotCount}).fill(6.6);
  console.log(`ckksPlain: ${ckksPlain.slice(0, 5)}...`);

  const ckksEncoded = ckksEncoder.encode(ckksPlain, Math.pow(2, 20));
  console.log(`ckksEncoded: ${ckksEncoded.save().slice(0, 50)}...`);

  ckksEncryptor.encrypt(ckksEncoded, ckksCipher);
  console.log("ckksCipher::parmsId: ", ckksCipher.parmsId.values);
  console.log("ckksCipher: ", ckksCipher.save().slice(0, 50));
  console.log("ckksCipher::scale: ", ckksCipher.scale);
  console.log("ckksCipher::isTransparent: ", ckksCipher.isTransparent);
  console.log("ckksCipher::isNttForm: ", ckksCipher.isNttForm);
})();

// Ref: https://github.com/s0l0ist/node-seal/blob/9618aca13e745ebb2a9c7d3a6b18d78e68d4aab9/src/__tests__/cipher-text.test.ts