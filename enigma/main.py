# This project is a simple implementation of the Enigma machine used by the Germans during World War II.

class Plugboard:
  def __init__(self, plugs: list[str]):
    self.left = []
    self.right = []
    for plug in plugs:
      self.left.append(plug[0])
      self.right.append(plug[1])

  def forward(self, c: str) -> str:
    if c in self.left:
      return self.right[self.left.index(c)]
    if c in self.right:
      return self.left[self.right.index(c)]
    return c
  
  def backward(self, c: str) -> str:
    return self.forward(c)

pb = Plugboard(['AR', 'GK', 'OX'])

class Reflector:
    def __init__(self, wiring: str):
        self.mapping = list(wiring)

    def reflect(self, c: str) -> str:
        idx = ord(c) - ord('A')
        return self.mapping[idx]

B = Reflector('YRUHQSLDPXNGOKMIEBFZCWVJAT')

class Rotor:
    def __init__(self, wiring: str, notch: str):
        self.left = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.right = list(wiring)
        self.notch = (ord(notch) - ord('A')) % 26
        self.position = 0

    def at_notch(self) -> bool:
        return self.position == self.notch

    def forward(self, c: str) -> str:
        c_index = (ord(c) - ord('A') + self.position) % 26
        mapped_char = self.right[c_index]
        mapped_index = (ord(mapped_char) - ord('A') - self.position) % 26
        return chr(mapped_index + ord('A'))

    def backward(self, c: str) -> str:
        # Convert character to 0-25 index and apply rotor position offset
        c_index = (ord(c) - ord('A') + self.position) % 26
        # Find the index in wiring that maps to c_index
        mapped_index = (self.right.index(chr(c_index + ord('A'))) - self.position) % 26
        return chr(mapped_index + ord('A'))

    def rotate(self):
        self.position = (self.position + 1) % 26

    def rotate_to(self, position: int):
        self.position = position

    def reached_notch(self) -> bool:
        return self.left[self.position] == self.notch

I   = Rotor('EKMFLGDQVZNTOWYHXUSPAIBRCJ', 'Q')
II  = Rotor('AJDKSIRUXBLHWTMCQGZNPYFVOE', 'E')
III = Rotor('BDFHJLCPRTXVZNYEIWGAKMUSQO', 'V')

class Enigma:
  def __init__(self, reflector: Reflector, rotor1: Rotor, rotor2: Rotor, rotor3: Rotor, plugboard: Plugboard):
    self.reflector = reflector
    self.rotors = [rotor1, rotor2, rotor3]
    self.plugboard = plugboard

  def set_rotors(self, positions: list[int]):
    for i, position in enumerate(positions):
      self.rotors[i].rotate_to(position)

  def step_rotors(self):
      rotor1, rotor2, rotor3 = self.rotors

      m_at_notch = rotor2.at_notch()
      r_at_notch = rotor3.at_notch()

      rotor3.rotate()

      if r_at_notch or rotor3.at_notch():
          rotor2.rotate()
          if m_at_notch:
              rotor1.rotate()

  def encipher(self, c: str) -> str:
      self.step_rotors()
      c = self.plugboard.forward(c)
      for rotor in reversed(self.rotors):
          c = rotor.forward(c)

      c = self.reflector.reflect(c)

      for rotor in self.rotors:
          c = rotor.backward(c)

      c = self.plugboard.backward(c)
      return c

enigma = Enigma(B, I, II, III, pb)
enigma.set_rotors([0, 1, 2])
original = 'TESTANDO'
criptografado = ''
for c in original:
   criptografado += enigma.encipher(c)
print(criptografado)