def fmix64(h):
    h &= 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    return h & 0xFFFFFFFFFFFFFFFF

class StandaloneX:
    def __init__(self, seed):
        self.data = []
        s = seed
        for _ in range(16): # 1024 bits
            s = (s + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
            self.data.append(fmix64(s))

    @staticmethod
    def bind(a_data, b_data):
        return [a_data[i] ^ b_data[i] for i in range(16)]

    @staticmethod
    def shift(data, n):
        n %= 1024
        giant = 0
        for i, word in enumerate(data): giant |= word << (64 * i)
        rotated = ((giant << n) | (giant >> (1024 - n))) & ((1 << 1024) - 1)
        return [(rotated >> (64 * i)) & 0xFFFFFFFFFFFFFFFF for i in range(16)]

    def get_element(self, r, c, m_data=None):
        m_data = m_data or self.data
        row_seed = fmix64(r)
        col_seed = fmix64(c + 0xABCDE)
        
        # Interaction manifold
        row_vec = [fmix64(row_seed + i) for i in range(16)]
        col_vec = [fmix64(col_seed + i) for i in range(16)]
        interaction = [row_vec[i] ^ col_vec[i] for i in range(16)]
        
        # Resolve bit 0
        res = m_data[0] ^ interaction[0]
        return 1 if (res & 1) == 0 else -1

if __name__ == "__main__":
    X = StandaloneX(seed=1)
    Y = StandaloneX(seed=2)
    # Symbolic XOR Composition
    Z_data = StandaloneX.bind(X.data, StandaloneX.shift(Y.data, 7))
    print(f"Resolved Z[0,0]: {X.get_element(0, 0, Z_data)}")

