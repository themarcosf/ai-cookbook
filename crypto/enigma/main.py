# This project is a simple implementation of the Enigma machine used by the Germans during World War II.

from plugboard import Plugboard
from reflector import Reflector
from rotor import Rotor
from enigma import Enigma

pb = Plugboard(['AR', 'GK', 'OX'])

B = Reflector('YRUHQSLDPXNGOKMIEBFZCWVJAT')

I   = Rotor('EKMFLGDQVZNTOWYHXUSPAIBRCJ', 'Q')
II  = Rotor('AJDKSIRUXBLHWTMCQGZNPYFVOE', 'E')
III = Rotor('BDFHJLCPRTXVZNYEIWGAKMUSQO', 'V')

enigma = Enigma(B, I, II, III, pb)
enigma.set_rotors([0, 1, 2])
original = 'TESTANDO'
criptografado = ''
for c in original:
   criptografado += enigma.encipher(c)
print(criptografado)