import SEAL from "node-seal";

(async () => {
  const seal = await SEAL();

  const polyModulusDegree = 4096;
  const bitSize = 20;

  const bfvCoeffModulus = seal.CoeffModulus.BFVDefault(polyModulusDegree);
  const bfvPlainModulus = seal.PlainModulus.Batching(polyModulusDegree, bitSize);
  const bfvEncParms = seal.EncryptionParameters(seal.SchemeType.bfv);
  bfvEncParms.setPolyModulusDegree(polyModulusDegree);
  bfvEncParms.setCoeffModulus(bfvCoeffModulus);
  bfvEncParms.setPlainModulus(bfvPlainModulus);
  const bfvContext = seal.Context(bfvEncParms);
  const bfvBatchEncoder = seal.BatchEncoder(bfvContext);
  const bfvKeyGenerator = seal.KeyGenerator(bfvContext);
  const bfvPublicKey = bfvKeyGenerator.createPublicKey();
  const bfvSecretKey = bfvKeyGenerator.secretKey();
  const bfvRelinearizationKeys = bfvKeyGenerator.createRelinKeys();
  const bfvEncryptor = seal.Encryptor(bfvContext, bfvPublicKey);
  const bfvDecryptor = seal.Decryptor(bfvContext, bfvSecretKey);

  const bfvEvaluator = seal.Evaluator(bfvContext);

  const plainOfNegatives = Int32Array.from({ length: bfvBatchEncoder.slotCount }, (_, i) => -i);
  const encodedNegatives = bfvBatchEncoder.encode(plainOfNegatives);
  const encryptedNegatives = bfvEncryptor.encrypt(encodedNegatives);
  console.log(`plainOfNegatives: ${plainOfNegatives.slice(0, 20)}...`);
  console.log(`encoder::encode: ${encodedNegatives.save().slice(0, 50)}...`);
  console.log(`encryptor::encrypt: ${encryptedNegatives.save().slice(0, 50)}...`);
  console.log("\n");

  const plainOfFives = Int32Array.from({ length: bfvBatchEncoder.slotCount }).fill(-5);
  const encodedFives = bfvBatchEncoder.encode(plainOfFives);
  const encryptedFives = bfvEncryptor.encrypt(encodedFives);
  console.log(`plainOfFives: ${plainOfFives.slice(0, 20)}...`);
  console.log(`encoder::encode: ${encodedFives.save().slice(0, 50)}...`);
  console.log(`encryptor::encrypt: ${encryptedFives.save().slice(0, 50)}...`);
  console.log("\n");

  const plainOfEvenNegatives = Int32Array.from({ length: bfvBatchEncoder.slotCount }, (_, i) => -2 * i);
  const encodedSub = bfvBatchEncoder.encode(plainOfEvenNegatives);
  const encryptedSub = bfvEncryptor.encrypt(encodedSub);
  console.log(`plainOfEvenNegatives: ${plainOfEvenNegatives.slice(0, 20)}...`);
  console.log(`encoder::encode: ${encodedSub.save().slice(0, 50)}...`);
  console.log(`encryptor::encrypt: ${encryptedSub.save().slice(0, 50)}...`);
  console.log("\n");

  const negated = bfvEvaluator.negate(encryptedNegatives);
  const decryptedNegated = bfvDecryptor.decrypt(negated);
  const decodedNegated = bfvBatchEncoder.decode(decryptedNegated, true);
  console.log(`evaluator::negate: ${negated.save().slice(0, 50)}...`);
  console.log(`decryptor::decrypt: ${decryptedNegated.save().slice(0, 50)}...`);
  console.log(`decoder::decode: ${decodedNegated.slice(0, 20)}...`);
  console.log(`decoded negated plain: ${decodedNegated.every((v, i) => -v === plainOfNegatives[i])}`);
  console.log("\n");

  const added = bfvEvaluator.add(encryptedNegatives, encryptedNegatives);
  const decryptedAdded = bfvDecryptor.decrypt(added);
  const decodedAdded = bfvBatchEncoder.decode(decryptedAdded, true);
  console.log(`evaluator::add: ${added.save().slice(0, 50)}...`);
  console.log(`decryptor::decrypt: ${decryptedAdded.save().slice(0, 50)}...`);
  console.log(`decoder::decode: ${decodedAdded.slice(0, 20)}...`);
  console.log(`decoded added plain: ${decodedAdded.every((v, i) => v === plainOfNegatives[i] * 2)}`);
  console.log("\n");

  const subtracted = bfvEvaluator.sub(encryptedNegatives, encryptedSub);
  const decryptedSubtracted = bfvDecryptor.decrypt(subtracted);
  const decodedSubtracted = bfvBatchEncoder.decode(decryptedSubtracted, true);
  console.log(`evaluator::sub: ${subtracted.save().slice(0, 50)}...`);
  console.log(`decryptor::decrypt: ${decryptedSubtracted.save().slice(0, 50)}...`);
  console.log(`decoder::decode: ${decodedSubtracted.slice(0, 20)}...`);
  console.log(`decoded subtracted evenPlain: ${decodedSubtracted.every((v, i) => v === plainOfNegatives[i] + 2 * i)}`);
  console.log("\n");

  const multipliedFail = bfvEvaluator.multiply(encryptedNegatives, encryptedNegatives);
  console.log(`[FAIL] evaluator::multiply: ${multipliedFail.save().slice(0, 50)}...`);

  const decryptedMultipliedFail = bfvDecryptor.decrypt(multipliedFail);
  console.log(`[FAIL] decryptor::decrypt: ${decryptedMultipliedFail.save().slice(0, 50)}...`);

  const decodedMultipliedFail = bfvBatchEncoder.decode(decryptedMultipliedFail, true);
  console.log(`[FAIL] decoded multiplied plain: ${decodedMultipliedFail.every((v, i) => {
    if (v !== plainOfNegatives[i] * plainOfNegatives[i]) {
      console.log(`[FAIL] decoded multiplied plain: ${v} !== ${plainOfNegatives[i]} * ${plainOfNegatives[i]}`);
      return false;
    }
    return true;
  })}`);
  console.log("\n");

  const multiplied = bfvEvaluator.multiply(encryptedFives, encryptedFives);
  const decryptedMultiplied = bfvDecryptor.decrypt(multiplied);
  const decodedMultiplied = bfvBatchEncoder.decode(decryptedMultiplied, true);
  console.log(`evaluator::multiply: ${multiplied.save().slice(0, 50)}...`);
  console.log(`decryptor::decrypt: ${decryptedMultiplied.save().slice(0, 50)}...`);
  console.log(`decoder::decode: ${decodedMultiplied.slice(0, 20)}...`);
  console.log(`decoded multiplied fives: ${decodedMultiplied.every((v, i) => v === plainOfFives[i] * plainOfFives[i])}`);
  console.log("\n");

  const squared = bfvEvaluator.square(encryptedFives);
  const decryptedSquared = bfvDecryptor.decrypt(squared);
  const decodedSquared = bfvBatchEncoder.decode(decryptedSquared, true);
  console.log(`evaluator::square: ${squared.save().slice(0, 50)}...`);
  console.log(`decryptor::decrypt: ${decryptedSquared.save().slice(0, 50)}...`);
  console.log(`decoder::decode: ${decodedSquared.slice(0, 20)}...`);
  console.log(`decoded squared fives: ${decodedSquared.every((v, i) => v === plainOfFives[i] * plainOfFives[i])}`);
  console.log("\n");

  const relinearized = bfvEvaluator.relinearize(squared, bfvRelinearizationKeys);
  const decryptedRelinearized = bfvDecryptor.decrypt(relinearized);
  const decodedRelinearized = bfvBatchEncoder.decode(decryptedRelinearized, true);
  console.log(`evaluator::relinearize: ${relinearized.save().slice(0, 50)}...`);
  console.log(`evaluator::relinearize === evaluator::squared?: ${relinearized.save() === squared.save()}`);
  console.log(`sizeof squared: ${squared.save().length}`);
  console.log(`sizeof relinearized: ${relinearized.save().length}`);
  console.log(`decryptor::decrypt: ${decryptedRelinearized.save().slice(0, 50)}...`);
  console.log(`decoder::decode: ${decodedRelinearized.slice(0, 20)}...`);
  console.log(`decoded relinearized plain: ${decodedRelinearized.every((v, i) => v === plainOfFives[i] * plainOfFives[i])}`);
  console.log("\n");
})();

// Ref: https://github.com/s0l0ist/node-seal/blob/9618aca13e745ebb2a9c7d3a6b18d78e68d4aab9/src/__tests__/evaluator.test.ts