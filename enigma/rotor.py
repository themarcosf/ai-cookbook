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
        c_index = (ord(c) - ord('A') + self.position) % 26
        mapped_index = (self.right.index(chr(c_index + ord('A'))) - self.position) % 26
        return chr(mapped_index + ord('A'))

    def rotate(self):
        self.position = (self.position + 1) % 26

    def rotate_to(self, position: int):
        self.position = position

    def reached_notch(self) -> bool:
        return self.left[self.position] == self.notch