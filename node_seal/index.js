///////////////////////////////////////////////////////////////////////////////
// This is intended to be a very simple demonstration of a homomorphic encryption
// input/output pipeline using SEAL to be integrated with a cryptographic neural
// network.
//
// We are currently implementing the complete pipeline in Rust, in order to achieve
// maximum performance, parallelism, and security. Our goal is to provide a fully
// functional pipeline so clients can run our models on their own data, without
// compromising their privacy.
//
// The example below demonstrates our approach for a single token. In a real-world
// scenario, we would need to handle multiple tokens. For example, DeepSeek-V2's
// tokenizer contains 100K tokens. In a `naive` calculation for an unknown sequence
// length the total number of possible sequences is 100K^n, where n is the sequence
// length. For a 10-token sequence length, this results in 100K^10 = (10^5)^10 = 10^50
// possible sequences. This should be adjusted for more realistic conditional
// probabilities, such as Markov chains or language grammars.
//
// As a final note, for this DEMO we departed from our underlying implementation
// in two ways:
// -1- we used the BFV algorithm instead of CKKS
// -2- we used two `theoretical` pub-secret key pairs and kept the noise param secret
//     instead of using a single pub-secret key pair, so the secret param can be
//     shared and randomized.
//
// source: https://huggingface.co/deepseek-ai/DeepSeek-V2
///////////////////////////////////////////////////////////////////////////////
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
  // Variables
  ///////////////////////////////////////////////////////////////////////////////

  // // Creating PlainText(s)
  // const plainA = seal.PlainText();
  // const plainB = seal.PlainText();

  // // Creating CipherText(s)
  // const cipherA = seal.CipherText();
  // const cipherB = seal.CipherText();

  // // Saving
  // // ... after some encoding...
  // const plainAbase64 = plainA.save(); // Saves as a base64 string.

  // // Loading. Create an empty instance, then use the following method
  // const uploadedPlain = seal.PlainText();
  // uploadedPlain.load(context, plainAbase64);

  // // Saving
  // // ... after some encryption...
  // const cipherAbase64 = cipherA.save(); // Saves as a base64 string.

  // // Loading. Create an empty instance, then use the following method
  // const uploadedCipherText = seal.CipherText();
  // uploadedCipherText.load(context, cipherAbase64);

  ///////////////////////////////////////////////////////////////////////////////
  // Simple inbound pipeline
  //
  // Public key: shared with anyone who wants to encrypt data
  // Secret key: only the provider has access
  // Secret noise param: only the client has access
  //
  // This example shows how to encrypt a token on the client side, send it to the
  // provider, and decrypt it on the provider side. The provider does not have
  // access to the secret noise param and thus cannot decode the original token
  // value.
  ///////////////////////////////////////////////////////////////////////////////

  // secret random noise parameter -- client side
  // this value can be as large as desired for security purposes
  const noiseParam = Int32Array.from([11]);
  console.log("Noise: ", noiseParam);

  // input plain data -- client side
  // assume we are operating on a single token
  const plainToken = Int32Array.from([42]);
  console.log("Plain token: ", plainToken);

  // input encoding and encryption -- client side
  const noise = encryptor.encrypt(encoder.encode(noiseParam));
  const encodedInput = encryptor.encrypt(encoder.encode(plainToken));
  const transitInputToken = evaluator.multiply(encodedInput, noise);
  console.log(
    "Last 20 digits of transitInputToken: ...",
    transitInputToken.save().slice(-20)
  );

  // input decryption and decoding -- provider side
  // note that the provider does not have access to the secret noise param
  // and thus cannot decrypt the original token value
  const decoded = encoder.decode(decryptor.decrypt(transitInputToken));
  const secureToken = decoded.filter((n) => n !== 0);
  console.log("Secure input token: ", secureToken);

  ///////////////////////////////////////////////////////////////////////////////
  // 3) Simple outbound pipeline
  //
  // Public key: only the provider has access
  // Secret key: only the client has access
  // Secret noise param: we could add this to the provider, but we don't need to.
  //
  // This is similar to the inbound pipeline, but the provider encrypts the output
  // token and sends it to the client. The client decrypts the token using the secret
  // key. The provider has no prior knowledge of the token mapping and thus cannot
  // decrypt the real token value.
  ///////////////////////////////////////////////////////////////////////////////
  // output secure token -- provider side
  const secureOutputToken = Int32Array.from([100]);
  console.log("Secure output token: ", secureOutputToken);

  // output encoding and encryption -- provider side
  const transitOutputToken = encryptor.encrypt(
    encoder.encode(secureOutputToken)
  );
  console.log(
    "Last 20 digits of transitOutputToken: ...",
    transitOutputToken.save().slice(-20)
  );

  // output decryption and decoding -- client side
  const decodedOutput = encoder.decode(decryptor.decrypt(transitOutputToken));
  const outputToken = decodedOutput.filter((n) => n !== 0);
  console.log("Decoded output token: ", outputToken);

  // Remapping output token -- client side
  // This is a simple example of remapping the output token. In a real-world
  // scenario, the remapping function would depend on the specific use case.
  const remappedOutputToken = outputToken.map((n) => n - 49);
  console.log("Remapped output token: ", remappedOutputToken);

  ///////////////////////////////////////////////////////////////////////////////
  // timing end and total time
  ///////////////////////////////////////////////////////////////////////////////
  const endTime = performance.now();
  console.log(`Total time: ${(endTime - startTime).toFixed(2)} ms`);
})();
