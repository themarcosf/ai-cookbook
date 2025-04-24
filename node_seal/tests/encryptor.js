import SEAL from "node-seal";

(async () => {
  const seal = await SEAL();

  const polyModulusDegree = 4096;
  const bitSize = 20;

  const coeffModulus = seal.CoeffModulus.BFVDefault(polyModulusDegree);
  const plainModulus = seal.PlainModulus.Batching(polyModulusDegree, bitSize);
  const bfvEncParms = seal.EncryptionParameters(seal.SchemeType.bfv);
  bfvEncParms.setPolyModulusDegree(polyModulusDegree);
  bfvEncParms.setCoeffModulus(coeffModulus);
  bfvEncParms.setPlainModulus(plainModulus);
  const bfvContext = seal.Context(bfvEncParms);
  const bfvKeyGenerator = seal.KeyGenerator(bfvContext);
  const bfvSecretKey = bfvKeyGenerator.secretKey();
  const bfvPublicKey = bfvKeyGenerator.createPublicKey();

  const encoder = seal.BatchEncoder(bfvContext);
  const encryptor = seal.Encryptor(bfvContext, bfvPublicKey, bfvSecretKey);

  const plain = Int32Array.from({ length: encoder.slotCount}).fill(5);
  console.log(`plain: ${plain.slice(0, 20)}`);

  const encoded = encoder.encode(plain);
  console.log(`encoder::encode: ${encoded.save().slice(0, 50)}...`);

  const encrypted = encryptor.encrypt(encoded);
  console.log(`encryptor::encrypt: ${encrypted.save().slice(0, 50)}...`);

  const serializable = encryptor.encryptSerializable(encoded);
  console.log(`encryptor::encryptSerializable: ${serializable.save().slice(0, 50)}...`);

  const symmetricEncryption = encryptor.encryptSymmetric(encoded);
  console.log(`encryptor::encryptSymmetric: ${symmetricEncryption.save().slice(0, 50)}...`);

  const serializableSymmetric = encryptor.encryptSymmetricSerializable(encoded);
  console.log(`encryptor::encryptSymmetricSerializable: ${serializableSymmetric.save().slice(0, 50)}...`);
})();

// Ref: https://github.com/s0l0ist/node-seal/blob/9618aca13e745ebb2a9c7d3a6b18d78e68d4aab9/src/__tests__/encryptor.test.ts