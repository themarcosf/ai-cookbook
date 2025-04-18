import * as fs from "fs";

import SEAL from "node-seal";

(async () => {
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
  const keyGenerator = seal.KeyGenerator(context);
  const secretKey = keyGenerator.secretKey();
  const publicKey = keyGenerator.createPublicKey();
  const secretBase64Key = secretKey.save();
  const publicBase64Key = publicKey.save();

  // save keys to current directory
  fs.writeFileSync("./keys/secretKey.txt", secretBase64Key);
  fs.writeFileSync("./keys/publicKey.txt", publicBase64Key);
})();
