"""
G-тетрада с K⁺ — структурный элемент теломерного G-квадруплекса.

Считаем две величины (обе с противовесной поправкой BSSE):
  1) образование тетрады из 4 гуанинов (8 водородных связей Хугстина):
       ΔE_tetrad = E_T(T) − 4·E_G(T)              (по C4-симметрии мономеры эквивалентны)
  2) связывание K⁺ тетрадой:
       ΔE_K = E_TK(TK) − E_T(TK) − E_K(TK)

Классика: RHF и MP2 (density fitting) в PySCF.
Квантовая часть: CAS(ne,no) вокруг уровня Ферми. Тетрада + K⁺ — 65 атомов,
~250 базисных функций: полные 4-индексные интегралы через PySCFDriver
непрактичны, поэтому активный гамильтониан берём из PySCF (CASCI h1eff/h2eff,
энергия ядра входит в константу) и отдаём в qiskit-nature как
ElectronicEnergy.from_raw_integrals -> ParityMapper -> VQE(UCCSD) / точно.
Для K⁺ (замкнутая оболочка [Ar]) корреляция активного пространства = 0,
его энергия в CAS-балансе — HF.
"""
import argparse
import time

import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.mp.dfmp2 import DFMP2

from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

from tetrad_geometry import build_tetrad

HA2KCAL = 627.5095


def atoms(syms, xyz, ghost_mask=None):
    ghost_mask = ghost_mask if ghost_mask is not None else [False] * len(syms)
    return [(("ghost-" if g else "") + s, tuple(p)) for s, p, g in zip(syms, xyz, ghost_mask)]


def run_hf(atom, basis, charge):
    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=0, verbose=0, max_memory=16000)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-9
    mf.kernel()
    assert mf.converged, "HF не сошёлся"
    return mf


def run_mp2(mf):
    return DFMP2(mf).run().e_tot


def k_weight(mf, mo_idx):
    """Доля орбиталей на атоме K (Малликен) — видно, попал ли K⁺ в активное пространство."""
    mol = mf.mol
    k_ao = [i for i, lab in enumerate(mol.ao_labels()) if lab.split()[1] == "K"]
    if not k_ao:
        return [0.0] * len(mo_idx)
    s = mf.get_ovlp()
    out = []
    for m in mo_idx:
        c = mf.mo_coeff[:, m]
        out.append(float(c[k_ao] @ (s @ c)[k_ao]))
    return out


def cas_qiskit(mf, ne, no, use_vqe):
    mc = mcscf.CASCI(mf, no, ne)
    h1, ecore = mc.get_h1eff()
    h2 = ao2mo.restore(1, mc.get_h2eff(), no)
    ham = ElectronicEnergy.from_raw_integrals(h1, h2)
    ham.constants["core"] = ecore
    prob = ElectronicStructureProblem(ham)
    prob.num_particles = (ne // 2, ne // 2)
    prob.num_spatial_orbitals = no
    mapper = ParityMapper(num_particles=prob.num_particles)
    nq = mapper.map(ham.second_q_op()).num_qubits

    e_exact = GroundStateEigensolver(mapper, NumPyMinimumEigensolver()).solve(prob).total_energies[0]
    e_ref = mc.kernel()[0]  # контроль: CASCI PySCF
    assert abs(e_exact - e_ref) < 1e-7, (e_exact, e_ref)

    e_vqe = None
    if use_vqe:
        hf = HartreeFock(no, prob.num_particles, mapper)
        ans = UCCSD(no, prob.num_particles, mapper, initial_state=hf)
        vqe = VQE(StatevectorEstimator(), ans, SLSQP(maxiter=300))
        vqe.initial_point = np.zeros(ans.num_parameters)
        e_vqe = GroundStateEigensolver(mapper, vqe).solve(prob).total_energies[0]
    nocc = mf.mol.nelectron // 2
    act = list(range(nocc - ne // 2, nocc - ne // 2 + no))
    return e_exact, e_vqe, nq, k_weight(mf, act)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="sto3g")
    ap.add_argument("--ne", type=int, default=4)
    ap.add_argument("--no", type=int, default=4)
    ap.add_argument("--no-vqe", action="store_true")
    args = ap.parse_args()
    vqe = not args.no_vqe

    syms, xyz, n = build_tetrad("auto")
    nt = 4 * n
    print(f"G-тетрада + K⁺: {len(syms)} атомов, базис {args.basis}, CAS({args.ne}e,{args.no}o)\n")

    calcs = {  # имя: (маска ghost, заряд)
        "TK": ([False] * len(syms), +1),
        "T(TK)": ([False] * nt + [True], 0),
        "K(TK)": ([True] * nt + [False], +1),
        "T": (None, 0),
        "G(T)": ([False] * n + [True] * (nt - n), 0),
    }
    E = {}
    for name, (mask, q) in calcs.items():
        t0 = time.time()
        if name in ("T", "G(T)"):
            at = atoms(syms[:nt], xyz[:nt], mask)
        else:
            at = atoms(syms, xyz, mask)
        mf = run_hf(at, args.basis, q)
        e = {"HF": mf.e_tot, "MP2": run_mp2(mf), "nao": mf.mol.nao}
        if name == "K(TK)":      # [Ar]-оболочка: корреляция активного пространства = 0
            e["CAS"] = e["VQE"] = mf.e_tot
        elif name in ("TK", "T(TK)"):
            ce, cv, nq, kw = cas_qiskit(mf, args.ne, args.no, vqe)
            e.update(CAS=ce, VQE=cv, nq=nq, kw=kw)
        # T и G(T) — только HF/MP2 (образование тетрады)
        E[name] = e
        extra = f", {e['nq']} кубитов, доля K в активных МО {np.round(e['kw'], 3).tolist()}" if "nq" in e else ""
        print(f"[{name:6}] {e['nao']} АО, {time.time() - t0:5.0f} с{extra}", flush=True)

    def d(key, a, b, c, mult=1):
        if any(E[x][key] is None for x in (a, b, c)):
            return None
        return (E[a][key] - E[b][key] - mult * E[c][key]) * HA2KCAL

    print("\nСвязывание K⁺ тетрадой, ΔE_K (ккал/моль, BSSE):")
    for key in ("HF", "MP2", "CAS", "VQE"):
        v = d(key, "TK", "T(TK)", "K(TK)")
        print(f"  {key:4} {v:9.2f}" if v is not None else f"  {key:4}      —")
    if vqe:
        print(f"  |VQE−CAS| комплекс = {abs(E['TK']['VQE'] - E['TK']['CAS']) * HA2KCAL:.2e}, "
              f"тетрада = {abs(E['T(TK)']['VQE'] - E['T(TK)']['CAS']) * HA2KCAL:.2e} ккал/моль")

    print("\nОбразование тетрады из 4 G, ΔE_tetrad (ккал/моль, BSSE):")
    for key in ("HF", "MP2"):
        v = (E["T"][key] - 4 * E["G(T)"][key]) * HA2KCAL
        print(f"  {key:4} {v:9.2f}   (на H-связь {v / 8:6.2f})")


if __name__ == "__main__":
    main()
