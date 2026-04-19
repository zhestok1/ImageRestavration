from abc import ABC, abstractmethod
import numpy as np

class Basis(ABC):
    """Абстрактный базовый класс"""

    def __init__(self, N):
        self.N = N

    @abstractmethod
    def basis_function(self, p, l):
        pass

    @abstractmethod
    def get_coeff(self, image, p, l):
        pass

    def get_n_coeffs(self, image, num_coeffs, order="zigzag"):
        coeffs = np.zeros((self.N, self.N), dtype=complex)

        count = 0
        for p, l in self._get_order(order):
            if count >= num_coeffs:
                break
            coeffs[p, l] = self.get_coeff(image, p, l)
            count += 1

        return coeffs

    @abstractmethod
    def recovery(self, coeffs):
        pass

    def _get_order(self, order_type):
        if order_type == "row":
            # Построчный порядок (слева направо, сверху вниз)
            for i in range(self.N):
                for j in range(self.N):
                    yield i, j

        elif order_type == "zigzag":
            for s in range(2 * self.N - 1):
                i_min = max(0, s - (self.N - 1))
                i_max = min(self.N - 1, s)

                if s % 2 == 0:
                    for i in range(i_min, i_max + 1):
                        yield i, s - i
                else:
                    for i in range(i_max, i_min - 1, -1):
                        yield i, s - i

        elif order_type == "hg":
            coords = []
            for n in range(self.N):
                for m in range(self.N):
                    coords.append((n, m, n + m))

            coords.sort(key=lambda x: (x[2], x[0]))

            for n, m, _ in coords:
                yield n, m

        elif order_type == "russian_doll":
            # Порядок "Матрёшка": рекурсивное вложение блоков
            coords = []
            block = self.N
            
            while block >= 1:
                for i in range(0, self.N, block):
                    for j in range(0, self.N, block):
                        if (i, j) not in coords:
                            coords.append((i, j))
                block //= 2
            
            for i, j in coords:
                yield i, j

        else:
            raise ValueError("Unknown order")
