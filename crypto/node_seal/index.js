import SEAL from "node-seal";

const seal = await SEAL();

////////////////////////
// Encryption Parameters
////////////////////////
const schemeType = seal.SchemeType.bfv;
const securityLevel = seal.SecurityLevel.tc128;
const polyModulusDegree = 4096;
const bitSizes = [36, 36, 37];
const bitSize = 20;

const encParms = seal.EncryptionParameters(schemeType);

// Set the PolyModulusDegree
encParms.setPolyModulusDegree(polyModulusDegree);

// Create a suitable set of CoeffModulus primes
encParms.setCoeffModulus(
  seal.CoeffModulus.Create(polyModulusDegree, Int32Array.from(bitSizes))
);

// Set the PlainModulus to a prime of bitSize 20.
encParms.setPlainModulus(
  seal.PlainModulus.Batching(polyModulusDegree, bitSize)
);

////////////////////////
// Context
////////////////////////
const context = seal.Context(encParms, true, securityLevel);

if (!context.parametersSet()) {
  throw new Error(
    "Could not set the parameters in the given context. Please try different encryption parameters."
  );
}

////////////////////////
// Keys
////////////////////////

// Create a new KeyGenerator (creates a new keypair internally)
const keyGenerator = seal.KeyGenerator(context)

const secretKey = keyGenerator.secretKey()
const publicKey = keyGenerator.createPublicKey()
const relinKey = keyGenerator.createRelinKeys()
// Generating Galois keys takes a while compared to the others
const galoisKey = keyGenerator.createGalisKeys()

// Saving a key to a string is the same for each type of key
const secretBase64Key = secretKey.save()
const publicBase64Key = publicKey.save()
const relinBase64Key = relinKey.save()
// Please note saving Galois keys can take an even longer time and the output is **very** large.
const galoisBase64Key = galoisKey.save()

// Loading a key from a base64 string is the same for each type of key
// Load from the base64 encoded string
const UploadedSecretKey = seal.SecretKey()
UploadedSecretKey.load(context, secretBase64Key)
...


// NOTE
//
// A KeyGenerator can also be instantiated with existing keys. This allows you to generate
// new Relin/Galois keys with a previously generated SecretKey.

// Uploading a SecretKey: first, create an Empty SecretKey to load
const UploadedSecretKey = seal.SecretKey()

// Load from the base64 encoded string
UploadedSecretKey.load(context, secretBase64Key)

// Create a new KeyGenerator (use uploaded secretKey)
const keyGenerator = seal.KeyGenerator(context, UploadedSecretKey)

// Similarly, you may also create a KeyGenerator with a PublicKey. However, the benefit is purley to
// save time by not generating a new PublicKey

// Uploading a PublicKey: first, create an Empty PublicKey to load
const UploadedPublicKey = seal.PublicKey()

// Load from the base64 encoded string
UploadedPublicKey.load(context, publicBase64Key)

// Create a new KeyGenerator (use both uploaded keys)
const keyGenerator = seal.KeyGenerator(context, UploadedSecretKey, UploadedPublicKey)

