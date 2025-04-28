import SEAL from "node-seal";

(async () => {
  const seal = await SEAL();
  const polyModulusDegree = 1024;
  const bitSizes = Int32Array.from([27]);
  const securityLevel = seal.SecurityLevel.tc128;
  
  const coeffModulus = seal.CoeffModulus.Create(polyModulusDegree, bitSizes);
  const encParms = seal.EncryptionParameters(seal.SchemeType.ckks);
  encParms.setPolyModulusDegree(polyModulusDegree);
  encParms.setCoeffModulus(coeffModulus);
  const context = seal.Context(encParms, true, securityLevel);

  const keyGenerator = seal.KeyGenerator(context)

  const secretKey = keyGenerator.secretKey()
  console.log(`secretKey: ${secretKey.save().slice(0, 50)}...`);

  const publicKey = keyGenerator.createPublicKey()
  console.log(`publicKey: ${publicKey.save().slice(0, 50)}...`);

  const ckksEncoder = seal.CKKSEncoder(context);
  console.log("ckksEncoder::slotCount: ", ckksEncoder.slotCount);
  console.log("\n");

  const raw = Float64Array.from({ length: ckksEncoder.slotCount }).map((_, i) => i == 0 ? 101 : 0)
  console.log(`raw: ${raw.slice(0, 10)}...`);

  const plain = ckksEncoder.encode(raw, Math.pow(2, 20));
  console.log(`plain: ${plain.save().slice(0, 50)}...`);

  const encryptor = seal.Encryptor(context, publicKey)
  const cipher = encryptor.encrypt(plain)
  console.log(`cipher: ${cipher.save().slice(0, 50)}...`);
  
  const decryptor = seal.Decryptor(context, secretKey)
  const decrypted = decryptor.decrypt(cipher)
  console.log(`decrypted: ${decrypted.save().slice(0, 50)}...`);

  const decoded = ckksEncoder.decode(decrypted);
  console.log(`decoded: ${decoded.slice(0, 10)}...`);

  const rounded = decoded.map((x) => Math.round(x));
  console.log(`rounded: ${rounded.slice(0, 10)}...`);
  console.log(`raw == rounded: ${raw.slice(0, 10).every((x, i) => x == rounded[i])}`);
})();

// Ref: https://github.com/s0l0ist/node-seal/tree/9618aca13e745ebb2a9c7d3a6b18d78e68d4aab9