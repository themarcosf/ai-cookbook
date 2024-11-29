from rotor import Rotor
from reflector import Reflector
from plugboard import Plugboard

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