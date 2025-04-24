import * as fs from "fs";

import SEAL from "node-seal";
import { performance } from "perf_hooks";

(async () => {
  ///////////////////////////////////////////////////////////////////////////////
  // Timing start
  ///////////////////////////////////////////////////////////////////////////////
  const startTime = performance.now();

  ///////////////////////////////////////////////////////////////////////////////
  // Encryption Parameters
  ///////////////////////////////////////////////////////////////////////////////
  const seal = await SEAL();

  // seal.SchemeType.bfv; seal.SchemeType.bgv; seal.SchemeType.ckks
  const schemeType = seal.SchemeType.bfv;

  // higher bit-strength options reduce homomorphic operations
  // seal.SecurityLevel.none; seal.SecurityLevel.[tc128, tc192, tc256]
  const securityLevel = seal.SecurityLevel.tc128;

  // needs to be a power of 2
  const polyModulusDegree = 4096;
  const bitSizes = [36, 36, 37];
  const bitSize = 20;

  const parms = seal.EncryptionParameters(schemeType);

  parms.setPolyModulusDegree(polyModulusDegree);

  parms.setCoeffModulus(
    seal.CoeffModulus.Create(polyModulusDegree, Int32Array.from(bitSizes))
  );

  // not applicable to CKKS
  parms.setPlainModulus(seal.PlainModulus.Batching(polyModulusDegree, bitSize));

  ///////////////////////////////////////////////////////////////////////////////
  // Context - used to create all instances which execute within the same params
  ///////////////////////////////////////////////////////////////////////////////
  const context = seal.Context(parms, true, securityLevel);

  if (!context.parametersSet()) {
    throw new Error(
      "Could not set the parameters in the given context. Please try different encryption parameters."
    );
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Keys
  ///////////////////////////////////////////////////////////////////////////////
  const secretKeyFile = "./keys/secretKey.txt";
  const secretBase64Key = fs.readFileSync(secretKeyFile, "utf8");
  const secretKey = seal.SecretKey();
  secretKey.load(context, secretBase64Key);

  const publicKeyFile = "./keys/publicKey.txt";
  const publicBase64Key = fs.readFileSync(publicKeyFile, "utf8");
  const publicKey = seal.PublicKey();
  publicKey.load(context, publicBase64Key);

  seal.KeyGenerator(context, secretKey, publicKey);

  ///////////////////////////////////////////////////////////////////////////////
  // Instances
  ///////////////////////////////////////////////////////////////////////////////
  const evaluator = seal.Evaluator(context); // used to perform homomorphic operations
  const encoder = seal.BatchEncoder(context); // used to encode to/decode from plaintext
  const encryptor = seal.Encryptor(context, publicKey); // used to encrypt plaintext to ciphertext
  const decryptor = seal.Decryptor(context, secretKey); // used to decrypt ciphertext to plaintext

  ///////////////////////////////////////////////////////////////////////////////
  // Homomorphic operations
  ///////////////////////////////////////////////////////////////////////////////
  const inputA = Int32Array.from([1, 2, 3, 4]);
  const inputB = Int32Array.from([5, 6, 7, 8]);
  console.log("Input A", inputA);
  console.log("Input B", inputB);

  const plainA = encoder.encode(inputA);
  const plainB = encoder.encode(inputB);

  const cipherA = encryptor.encrypt(plainA);
  const cipherB = encryptor.encrypt(plainB);

  const cipherAdd = evaluator.add(cipherA, cipherB);

  const plainAdd = decryptor.decrypt(cipherAdd);
  const decodedAdd = encoder.decode(plainAdd).filter((n) => n !== 0);
  console.log("Decoded addition:", decodedAdd);

  const cipherMult = evaluator.multiply(cipherA, cipherB);

  const plainMult = decryptor.decrypt(cipherMult);
  const decodedMult = encoder.decode(plainMult).filter((n) => n !== 0);
  console.log("Decoded multiplication:", decodedMult);

  ///////////////////////////////////////////////////////////////////////////////
  // timing end and total time
  ///////////////////////////////////////////////////////////////////////////////
  const endTime = performance.now();
  console.log(`Total time: ${(endTime - startTime).toFixed(2)} ms`);
})();
