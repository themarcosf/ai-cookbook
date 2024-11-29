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