import SEAL from "node-seal";

(async () => {
  const seal = await SEAL();

  const securityLevel = seal.SecurityLevel.tc128;
  const polyModulusDegree = 4096;
  const bitSizes = Int32Array.from([46, 16, 46])
  const bitSize = 20

  const coeffModulus = seal.CoeffModulus.Create(polyModulusDegree, bitSizes);
  const plainModulus = seal.PlainModulus.Batching(polyModulusDegree, bitSize);

  const bfvEncParms = seal.EncryptionParameters(seal.SchemeType.bfv);
  bfvEncParms.setPolyModulusDegree(polyModulusDegree);
  bfvEncParms.setCoeffModulus(coeffModulus);
  bfvEncParms.setPlainModulus(plainModulus);

  const bfvContext = seal.Context(bfvEncParms, true, securityLevel);
  console.log("bfvContext::parametersSet: ", bfvContext.parametersSet());

  const bfvBatchEncoder = seal.BatchEncoder(bfvContext);
  console.log(`bfvBatchEncoder::slotCount: ${bfvBatchEncoder.slotCount}`);

  const bfvKeyGenerator = seal.KeyGenerator(bfvContext);
  const bfvSecretKey = bfvKeyGenerator.secretKey();
  const bfvPublicKey = bfvKeyGenerator.createPublicKey();
  console.log(`bfvKeyGenerator::publicKey: ${bfvPublicKey.save().slice(0, 50)}...`);
  console.log(`bfvKeyGenerator::publicKey::length: ${bfvPublicKey.save().length}`);
  console.log(`bfvKeyGenerator::secretKey: ${bfvSecretKey.save().slice(0, 50)}...`);
  console.log(`bfvKeyGenerator::secretKey::length: ${bfvSecretKey.save().length}`);
  console.log("\n");
  
  const bfvEncryptor = seal.Encryptor(bfvContext, bfvPublicKey);
  
  const bfvDecryptor = seal.Decryptor(bfvContext, bfvSecretKey);
  
  const plain = Int32Array.from({ length: bfvBatchEncoder.slotCount }).fill(5);
  console.log(`plain: ${plain.slice(0, 20)}...`);
  
  const encoded = bfvBatchEncoder.encode(plain);
  console.log(`encoded: ${encoded.save().slice(0, 50)}...`);
  
  const encrypted = bfvEncryptor.encrypt(encoded);
  console.log(`encrypted: ${encrypted.save().slice(0, 50)}...`);
  console.log("\n");
  
  const decrypted = bfvDecryptor.decrypt(encrypted);
  console.log(`bfvDecryptor::invariantNoiseBudget: ${bfvDecryptor.invariantNoiseBudget(encrypted)}`);
  console.log(`decrypted: ${decrypted.save().slice(0, 50)}...`);

  const decoded = bfvBatchEncoder.decode(decrypted);
  console.log(`decoded: ${decoded.slice(0, 20)}...`);
  console.log(`decoded === plain: ${decoded.toString() === plain.toString()}`);
})();

// Ref: https://github.com/s0l0ist/node-seal/blob/9618aca13e745ebb2a9c7d3a6b18d78e68d4aab9/src/__tests__/decryptor.test.ts