from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from binascii import hexlify as hx
from os import urandom

# pick random 16-byte key using Python's crypto PRNG
key = urandom(16)
print(f"key: {hx(key)}")

# create AES cipher object
cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
encryptor = cipher.encryptor()
decryptor = cipher.decryptor()

# set plaintext block p to the all-zero string
p = b"\x00" * 16
print(f"p: {hx(p)}")

# encrypt p to get ciphertext block c
c_1 = encryptor.update(p)
c_2 = encryptor.update(p)
encryptor.finalize()
print(f"c_1: {hx(c_1)}")
print(f"c_2: {hx(c_2)}")

# decrypt c to get plaintext block p'
p_prime_1 = decryptor.update(c_1)
p_prime_2 = decryptor.update(c_2)
decryptor.finalize()
print(f"p_prime_1': {hx(p_prime_1)}")
print(f"p_prime_2': {hx(p_prime_2)}")