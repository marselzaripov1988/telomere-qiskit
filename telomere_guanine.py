"""
Квантово-химическая модель фрагмента теломеры (TTAGGG)n в qiskit-nature.

Полный нуклеотид / G-квадруплекс слишком велик для симуляции кубитов,
поэтому берём ключевой фрагмент — основание гуанин (G), которое образует
G-тетрады в теломерных G-квадруплексах, и считаем его в малом активном
пространстве вокруг HOMO/LUMO (π-орбитали, ответственные за стэкинг и
окисление G — типичное место повреждения теломер).

Пайплайн:
  SMILES -> 3D (RDKit, MMFF) -> HF (PySCF) -> активное пространство
  -> фермионный гамильтониан -> кубиты (ParityMapper) -> VQE(UCCSD)
  и точная диагонализация для проверки.
"""
import argparse
import time

import numpy as np
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

# Основания теломерного повтора TTAGGG (9H-/1H-таутомеры, без сахара)
BASES = {
    "G": "Nc1nc2[nH]cnc2c(=O)[nH]1",   # гуанин
    "A": "Nc1ncnc2[nH]cnc12",          # аденин
    "T": "Cc1c[nH]c(=O)[nH]c1=O",      # тимин
}


def build_geometry(smiles: str, seed: int = 42) -> str:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    lines = []
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}")
    return "; ".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="G", choices=BASES)
    ap.add_argument("--basis", default="sto3g")
    ap.add_argument("--ne", type=int, default=4, help="активных электронов")
    ap.add_argument("--no", type=int, default=4, help="активных орбиталей")
    args = ap.parse_args()

    geom = build_geometry(BASES[args.base])
    print(f"Основание {args.base}: {geom.count(';') + 1} атомов, базис {args.basis}")

    t0 = time.time()
    driver = PySCFDriver(atom=geom, basis=args.basis, charge=0, spin=0,
                         unit=DistanceUnit.ANGSTROM)
    problem = driver.run()
    print(f"HF готов за {time.time() - t0:.1f} с: "
          f"{problem.num_spatial_orbitals} орбиталей, "
          f"{sum(problem.num_particles)} электронов")

    problem = ActiveSpaceTransformer(args.ne, args.no).transform(problem)
    mapper = ParityMapper(num_particles=problem.num_particles)  # -2 кубита
    qubit_op = mapper.map(problem.hamiltonian.second_q_op())
    print(f"Активное пространство ({args.ne}e, {args.no}o) -> "
          f"{qubit_op.num_qubits} кубитов, {len(qubit_op)} паули-слагаемых")

    # Точный эталон
    exact = GroundStateEigensolver(mapper, NumPyMinimumEigensolver()).solve(problem)
    e_exact = exact.total_energies[0]

    # VQE с анзацем UCCSD поверх Хартри–Фока
    hf = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    ansatz = UCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper,
                   initial_state=hf)
    vqe = VQE(StatevectorEstimator(), ansatz, SLSQP(maxiter=300))
    vqe.initial_point = np.zeros(ansatz.num_parameters)
    t0 = time.time()
    res = GroundStateEigensolver(mapper, vqe).solve(problem)
    e_vqe = res.total_energies[0]

    e_hf = problem.reference_energy
    print(f"\nUCCSD: {ansatz.num_parameters} параметров, VQE {time.time() - t0:.1f} с")
    print(f"E(HF)    = {e_hf:.8f} Ha")
    print(f"E(VQE)   = {e_vqe:.8f} Ha")
    print(f"E(exact) = {e_exact:.8f} Ha  (CASCI в активном пространстве)")
    print(f"Корреляция = {(e_exact - e_hf) * 1000:.3f} mHa, "
          f"|VQE-exact| = {abs(e_vqe - e_exact) * 1000:.2e} mHa")


if __name__ == "__main__":
    main()
