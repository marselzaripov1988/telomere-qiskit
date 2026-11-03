"""
Стэкинг G–G — два соседних гуанина теломерной цепочки (TTAGGG).

Геометрия: плоский гуанин (RDKit/MMFF), копия сдвинута на d Å по нормали
и повёрнута на 36° (идеальная B-ДНК: rise 3.38 Å, twist 36°).

Энергия взаимодействия с противовесной поправкой BSSE (Boys–Bernardi):
    ΔE(d) = E_AB(AB) − E_A(AB) − E_B(AB)
где мономер считается в полном базисе димера (атомы партнёра — ghost).

Методы на каждой точке:
  HF                — без корреляции (стэкинг на HF отталкивательный);
  MP2               — классический эталон, ловит дисперсию всех электронов;
  CAS (exact / VQE) — квантовая часть: π-орбитали HOMO/LUMO,
                      димер (4e,4o) = 6 кубитов, мономер (2e,2o) = 2 кубита.
                      Размерно-согласовано: при d→∞ CAS(4,4) → 2×CAS(2,2).
"""
import argparse
import time

import numpy as np
from pyscf import gto, scf, mp
from rdkit import Chem
from rdkit.Chem import AllChem

from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

from telomere_guanine import BASES

HA2KCAL = 627.5095


def planar_monomer(smiles: str, seed: int = 42):
    """Координаты основания: центр масс в 0, плоскость кольца = xy."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    xyz = mol.GetConformer().GetPositions()
    sym = [a.GetSymbol() for a in mol.GetAtoms()]
    xyz -= xyz.mean(axis=0)
    # PCA: наименьшая главная ось — нормаль к плоскости -> z
    _, _, vt = np.linalg.svd(xyz)
    xyz = xyz @ vt.T
    xyz[:, 2] = 0.0  # гуанин плоский; убираем шум MMFF (NH2 почти в плоскости)
    return sym, xyz


def stacked_partner(xyz, d, twist_deg):
    t = np.radians(twist_deg)
    rot = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    out = xyz @ rot.T
    out[:, 2] += d
    return out


def atom_string(sym, xyz, ghost=False):
    pre = "ghost-" if ghost else ""
    return "; ".join(f"{pre}{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(sym, xyz))


def classical(atom, basis):
    m = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(m).run(conv_tol=1e-10)
    return mf.e_tot, mp.MP2(mf).run().e_tot


def cas_energy(atom, basis, ne, no, use_vqe=False):
    problem = PySCFDriver(atom=atom, basis=basis, unit=DistanceUnit.ANGSTROM).run()
    problem = ActiveSpaceTransformer(ne, no).transform(problem)
    mapper = ParityMapper(num_particles=problem.num_particles)
    nq = mapper.map(problem.hamiltonian.second_q_op()).num_qubits
    e_exact = GroundStateEigensolver(mapper, NumPyMinimumEigensolver()).solve(problem).total_energies[0]
    e_vqe = None
    if use_vqe:
        hf = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
        ans = UCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper, initial_state=hf)
        vqe = VQE(StatevectorEstimator(), ans, SLSQP(maxiter=300))
        vqe.initial_point = np.zeros(ans.num_parameters)
        e_vqe = GroundStateEigensolver(mapper, vqe).solve(problem).total_energies[0]
    return e_exact, e_vqe, nq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="sto3g")
    ap.add_argument("--twist", type=float, default=36.0)
    ap.add_argument("--dists", type=float, nargs="+", default=[3.0, 3.2, 3.4, 3.7, 4.0, 5.0])
    ap.add_argument("--vqe-at", type=float, default=3.4, help="на какой d прогнать VQE")
    args = ap.parse_args()

    sym, xa = planar_monomer(BASES["G"])
    print(f"G–G стэкинг: {2 * len(sym)} атомов, twist {args.twist}°, базис {args.basis}")
    print("ΔE в ккал/моль (BSSE-корр.). ΔCAS — вклад корреляции π-активного пространства.\n")
    print(f"{'d, Å':>5} {'ΔE(HF)':>9} {'ΔE(MP2)':>9} {'ΔE(CAS)':>9} {'ΔCAS':>8} {'ΔE(VQE)':>9}  t, c")

    rows = []
    for d in args.dists:
        t0 = time.time()
        xb = stacked_partner(xa, d, args.twist)
        ab = atom_string(sym, xa) + "; " + atom_string(sym, xb)
        a_g = atom_string(sym, xa) + "; " + atom_string(sym, xb, ghost=True)
        b_g = atom_string(sym, xa, ghost=True) + "; " + atom_string(sym, xb)

        (hf_ab, mp_ab), (hf_a, mp_a), (hf_b, mp_b) = (classical(x, args.basis) for x in (ab, a_g, b_g))
        vqe = abs(d - args.vqe_at) < 1e-6
        cas_ab, vqe_ab, nq_ab = cas_energy(ab, args.basis, 4, 4, vqe)
        cas_a, vqe_a, nq_a = cas_energy(a_g, args.basis, 2, 2, vqe)
        cas_b, vqe_b, _ = cas_energy(b_g, args.basis, 2, 2, vqe)

        d_hf = (hf_ab - hf_a - hf_b) * HA2KCAL
        d_mp2 = (mp_ab - mp_a - mp_b) * HA2KCAL
        d_cas = (cas_ab - cas_a - cas_b) * HA2KCAL
        d_vqe = (vqe_ab - vqe_a - vqe_b) * HA2KCAL if vqe else None
        rows.append((d, d_hf, d_mp2, d_cas))
        vqe_s = f"{d_vqe:9.3f}" if vqe else f"{'—':>9}"
        print(f"{d:5.2f} {d_hf:9.3f} {d_mp2:9.3f} {d_cas:9.3f} {d_cas - d_hf:8.3f} {vqe_s}  {time.time() - t0:5.0f}",
              flush=True)

    print(f"\nКубиты: димер {nq_ab}, мономер {nq_a}.")
    best = min(rows, key=lambda r: r[2])
    print(f"Минимум MP2: d = {best[0]:.2f} Å, ΔE = {best[2]:.2f} ккал/моль")


if __name__ == "__main__":
    main()
