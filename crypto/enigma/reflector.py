class Reflector:
    def __init__(self, wiring: str):
        self.mapping = list(wiring)

    def reflect(self, c: str) -> str:
        idx = ord(c) - ord('A')
        return self.mapping[idx]