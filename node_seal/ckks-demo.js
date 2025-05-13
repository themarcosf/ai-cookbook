import SEAL from "node-seal";

function getPolyCoeffs(ciphertext, size, coeffModulus) {
  console.log('\n    -> getPolyCoeffs');
  const modulus_value = coeffModulus[0];
  console.log(`modulus_value: ${modulus_value}`);
  return "pending"
}

function partialEval(ciphertext) {
  console.log('\n-> partialEval');
  const size = ciphertext.size
  console.log(`ciphertext_size: ${size}`);
  const polyModulusDegree = ciphertext.polyModulusDegree
  console.log(`polyModulusDegree: ${polyModulusDegree}`);
  const coeffModulusSize = ciphertext.coeffModulusSize
  console.log(`coeffModulusSize: ${coeffModulusSize}`);
  const wordCount = size * polyModulusDegree * coeffModulusSize
  console.log(`wordCount: ${wordCount}`);

  const contextData = this._parent.context.getContextData(ciphertext.parmsId)
  const parms = contextData.parms
  const coeffModulus = parms.coeffModulus
  const polyCoeffs = getPolyCoeffs(ciphertext, size, coeffModulus)
  console.log(`\n    ---> getPolyCoeffs: ${polyCoeffs}`);

  return 'pending'
}


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
  console.log(`publicKey: ${publicKey.save().slice(0, 50)}...\n`);

  const ckksEncoder = seal.CKKSEncoder(context);
  console.log(`ckksEncoder::slotCount: ${ckksEncoder.slotCount}\n`);

  const input = Float64Array.from({ length: ckksEncoder.slotCount }).map((_, i) => i+1)
  console.log(`input: ${input.slice(0, 10)}, ..., ${input.slice(-10)}`);
  const raw = Float64Array.from({ length: ckksEncoder.slotCount }).map((_, i) => input[i]/10)
  console.log(`raw: ${raw.slice(0, 10)}, ..., ${raw.slice(-10)}`);

  const plain = ckksEncoder.encode(raw, Math.pow(2, 20));
  console.log(`plain: ${plain.save().slice(0, 50)}...`);

  const encryptor = seal.Encryptor(context, publicKey)
  const cipher = encryptor.encrypt(plain)
  console.log(`cipher: ${cipher.save().slice(0, 50)}...`);

  const evaluator = seal.Evaluator(context)
  evaluator._parent = {seal, context};
  evaluator.partialEval = partialEval.bind(evaluator);

  const added = evaluator.add(cipher, cipher)
  console.log(`added: ${added.save().slice(0, 50)}...`);

  const partial = evaluator.partialEval(cipher)
  console.log(`\n---> partial: ${partial.slice(0, 20)}...\n`);
  
  const decryptor = seal.Decryptor(context, secretKey)
  const decrypted = decryptor.decrypt(added)
  console.log(`decrypted: ${decrypted.save().slice(0, 50)}...`);

  const decoded = ckksEncoder.decode(decrypted);
  console.log(`decoded: ${decoded.slice(0, 10)}...`);

  const rounded = decoded.map((x) => Math.round(x * 10));
  console.log(`rounded: ${rounded.slice(0, 10)}, ..., ${rounded.slice(-10)}`);
  console.log(`raw == rounded: ${input.slice(0, 10).every((x, i) => x * 2 == rounded[i])}`);
})();

// Ref: https://github.com/s0l0ist/node-seal/tree/9618aca13e745ebb2a9c7d3a6b18d78e68d4aab9