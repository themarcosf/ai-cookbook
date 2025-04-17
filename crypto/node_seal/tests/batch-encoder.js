import SEAL from "node-seal";

(async () => {
  const seal = await SEAL();

  const securityLevel = seal.SecurityLevel.tc128;
  const schemeType = seal.SchemeType.bfv;
  const polyModulusDegree = 1024;
  const bitSizes = Int32Array.from([27]);
  const bitSize = 20;

  const coeffModulus = seal.CoeffModulus.Create(polyModulusDegree, bitSizes);
  const plainModulus = seal.PlainModulus.Batching(polyModulusDegree, bitSize);
  const encParms = seal.EncryptionParameters(schemeType);
  encParms.setPolyModulusDegree(polyModulusDegree);
  encParms.setCoeffModulus(coeffModulus);
  encParms.setPlainModulus(plainModulus);
  const context = seal.Context(encParms, true, securityLevel);

  const batchEncoder = seal.BatchEncoder(context);
  console.log("batchEncoder::slotCount: ", batchEncoder.slotCount);
  console.log("\n");
  
  const int32plain = Int32Array.from({length: batchEncoder.slotCount}, (_, i) => -i);
  const int32encoded = batchEncoder.encode(int32plain);
  const int32decoded = batchEncoder.decode(int32encoded, true);
  console.log(`int32plain: ${int32plain.slice(0, 5)}...`);
  console.log(`int32encoded: ${int32encoded.save().slice(0, 50)}...`);
  console.log(`int32decoded: ${int32decoded.slice(0, 5)}...`);
  console.log("\n");
  
  const int64plain = BigInt64Array.from({length: batchEncoder.slotCount}, (_, i) => BigInt(-i));
  const int64encoded = batchEncoder.encode(int64plain);
  const int64decoded = batchEncoder.decode(int64encoded, true);
  console.log(`int64plain: ${int64plain.slice(0, 5)}...`);
  console.log(`int64encoded: ${int64encoded.save().slice(0, 50)}...`);
  console.log(`int64decoded: ${int64decoded.slice(0, 5)}...`);
  console.log("\n");
  
  const uint32plain = Uint32Array.from({length: batchEncoder.slotCount}, (_, i) => i);
  const uint32encoded = batchEncoder.encode(uint32plain);
  const uint32decoded = batchEncoder.decode(uint32encoded, true);
  console.log(`uint32plain: ${uint32plain.slice(0, 5)}...`);
  console.log(`uint32encoded: ${uint32encoded.save().slice(0, 50)}...`);
  console.log(`uint32decoded: ${uint32decoded.slice(0, 5)}...`);
  console.log("\n");

  const uint64plain = BigUint64Array.from({length: batchEncoder.slotCount}, (_, i) => BigInt(i));
  const uint64encoded = batchEncoder.encode(uint64plain);
  const uint64decoded = batchEncoder.decode(uint64encoded, true);
  console.log(`uint64plain: ${uint64plain.slice(0, 5)}...`);
  console.log(`uint64encoded: ${uint64encoded.save().slice(0, 50)}...`);
  console.log(`uint64decoded: ${uint64decoded.slice(0, 5)}...`);
})();

// Ref: https://github.com/s0l0ist/node-seal/blob/9618aca13e745ebb2a9c7d3a6b18d78e68d4aab9/src/__tests__/batch-encoder.test.ts