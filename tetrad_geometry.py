"""
Геометрия G-тетрады (G-квартета) с ионом K⁺.

Четыре гуанина связаны по Хугстину в плоский квадрат с симметрией C4:
Уотсон–Криковский край G_i (N1–H, N2–H) донирует водородные связи на
Хугстиновский край G_{i+1} (O6, N7). Атомы O6 смотрят в центр, где в
G-квадруплексе сидит K⁺ — между двумя тетрадами, т.е. на ~h Å над
плоскостью одной тетрады (K–O6 ≈ 2.8 Å).

Положение мономера (угол + сдвиг в плоскости) подбирается под целевые
длины H-связей: N1···O6' 2.85 Å, N2···N7' 2.90 Å, H линейно между ними.
"""
import numpy as np
from scipy.optimize import minimize

from stacking_gg import planar_monomer
from telomere_guanine import BASES

# индексы тяжёлых атомов в SMILES гуанина "Nc1nc2[nH]cnc2c(=O)[nH]1"
N2, N7, O6, N1 = 0, 6, 9, 10
TARGET = {"N1-O6": 2.85, "N2-N7": 2.90}
K_O6 = 2.80


def rot_z(t):
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


def _hydrogens(sym, xyz, heavy):
    return [i for i, s in enumerate(sym) if s == "H" and np.linalg.norm(xyz[i] - xyz[heavy]) < 1.15]


def build_tetrad(k_height=None):
    """Возвращает (symbols, xyz) тетрады; K⁺ добавляется, если k_height != None
    ('auto' — по расстоянию K–O6 = 2.8 Å)."""
    sym, g = planar_monomer(BASES["G"])
    h_n1 = _hydrogens(sym, g, N1)[0]
    h_n2 = _hydrogens(sym, g, N2)

    def place(p):
        return g @ rot_z(p[0]).T + np.array([p[1], p[2], 0.0])

    def cost(p, sign):
        a = place(p)
        b = a @ rot_z(sign * np.pi / 2).T  # соседний гуанин
        d = lambda x, y: np.linalg.norm(x - y)
        c = (d(a[N1], b[O6]) - TARGET["N1-O6"]) ** 2
        c += (d(a[N2], b[N7]) - TARGET["N2-N7"]) ** 2
        # линейность: донор–H–акцептор (H ближе к акцептору на длину N–H)
        c += (d(a[h_n1], b[O6]) - (TARGET["N1-O6"] - 1.01)) ** 2
        c += min((d(a[h], b[N7]) - (TARGET["N2-N7"] - 1.01)) ** 2 for h in h_n2)
        # без столкновений с другими тремя копиями
        for k in (1, 2, 3):
            other = a @ rot_z(k * np.pi / 2).T
            dist = np.linalg.norm(a[:, None] - other[None], axis=-1)
            c += np.sum(np.clip(1.6 - dist, 0, None) ** 2) * 10
        return c

    best = None
    for sign in (1, -1):
        for t0 in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            for r0 in (3.5, 5.0):
                res = minimize(cost, [t0, r0 * np.cos(t0), r0 * np.sin(t0)], args=(sign,),
                               method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 20000})
                if best is None or res.fun < best[0]:
                    best = (res.fun, res.x, sign)
    _, p, sign = best
    a = place(p)
    xyz = np.vstack([a @ rot_z(k * sign * np.pi / 2).T for k in range(4)])
    syms = sym * 4

    if k_height is not None:
        r_o6 = np.linalg.norm(a[O6, :2])
        h = np.sqrt(max(K_O6 ** 2 - r_o6 ** 2, 0.0)) if k_height == "auto" else float(k_height)
        xyz = np.vstack([xyz, [0.0, 0.0, h]])
        syms = syms + ["K"]
    return syms, xyz, len(sym)


def report(syms, xyz, n):
    a, b = xyz[:n], xyz[n:2 * n]
    d = lambda x, y: np.linalg.norm(x - y)
    print(f"N1···O6' = {d(a[N1], b[O6]):.3f} Å, N2···N7' = {d(a[N2], b[N7]):.3f} Å")
    print(f"O6–центр = {np.linalg.norm(a[O6, :2]):.3f} Å, O6···O6' = {d(a[O6], b[O6]):.3f} Å")
    heavy = [i for i, s in enumerate(syms[:4 * n]) if s != "H"]
    dm = np.linalg.norm(xyz[:4 * n, None] - xyz[None, :4 * n], axis=-1)
    inter = np.array([[i // n != j // n for j in range(4 * n)] for i in range(4 * n)])
    print(f"мин. межмолекулярное расстояние = {dm[inter].min():.3f} Å")
    if syms[-1] == "K":
        print(f"K⁺ над плоскостью h = {xyz[-1, 2]:.3f} Å, K–O6 = {d(xyz[-1], a[O6]):.3f} Å")


if __name__ == "__main__":
    s, x, n = build_tetrad("auto")
    report(s, x, n)
    with open("g_tetrad_K.xyz", "w") as f:
        f.write(f"{len(s)}\nG-tetrad + K+\n")
        f.writelines(f"{e} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n" for e, p in zip(s, x))
    print("записано g_tetrad_K.xyz")
