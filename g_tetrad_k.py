"""
G-тетрада с K⁺ — структурный элемент теломерного G-квадруплекса.

Считаем две величины (обе с противовесной поправкой BSSE):
  1) связывание K⁺ тетрадой:
       ΔE_K = E_TK(TK) − E_T(TK) − E_K(TK)
  2) образование тетрады из 4 гуанинов (8 водородных связей Хугстина):
       ΔE_tetrad = E_T(T) − 4·E_G(T)              (по C4-симметрии мономеры эквивалентны)

Классика: RHF и MP2 (density fitting) в PySCF.
Квантовая часть: активный гамильтониан берётся из PySCF (DF-CASCI h1eff/h2eff,
энергия ядра — в константе) и отдаётся в qiskit-nature через
ElectronicEnergy.from_raw_integrals -> ParityMapper -> VQE(UCCSD) / точно.

Выбор активного пространства (--active):
  fermi — ne/2 верхних занятых + остальные нижние виртуальные (HOMO/LUMO);
  o6k   — занятые МО с наибольшим весом на O6 (неподелённые пары, смотрящие на K⁺)
          + виртуальные с наибольшим весом на центре K (4s/4p). Описывает перенос
          заряда O6 → K⁺. Для T(TK) K — ghost, выбор по тем же правилам
          (базисные функции на центре K) — баланс сохраняется.
Для K⁺ ([Ar]) корреляция активного пространства = 0, в CAS-балансе — HF.
"""
import argparse
import json
import os
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

from tetrad_geometry import build_tetrad, O6

HA2KCAL = 627.5095


def atoms(syms, xyz, ghost_mask=None):
    ghost_mask = ghost_mask if ghost_mask is not None else [False] * len(syms)
    return [(("ghost-" if g else "") + s, tuple(p)) for s, p, g in zip(syms, xyz, ghost_mask)]


def run_hf(atom, basis, charge, chk):
    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=0, verbose=0, max_memory=18000)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-9
    mf.chkfile = chk
    if os.path.exists(chk):  # повторный запуск: старт с сохранённой плотности
        mf.init_guess = "chkfile"
    mf.kernel()
    assert mf.converged, "HF не сошёлся"
    return mf


def atom_weights(mf, atom_ids):
    """Малликеновский вес каждой МО на заданных атомах (ghost-центры тоже считаются)."""
    if not atom_ids:
        return np.zeros(mf.mo_coeff.shape[1])
    sl = mf.mol.aoslice_by_atom()
    ao = np.concatenate([np.arange(sl[a, 2], sl[a, 3]) for a in atom_ids])
    c = mf.mo_coeff
    sc = mf.get_ovlp() @ c
    return np.einsum("ip,ip->p", c[ao], sc[ao])


def shells(mos, energies, tol=1e-4):
    """Группирует МО в вырожденные уровни (по C4: A/B — 1 МО, E — 2 МО)."""
    out = []
    for m in sorted(mos, key=lambda i: energies[i]):
        if out and abs(energies[m] - energies[out[-1][-1]]) < tol:
            out[-1].append(m)
        else:
            out.append([m])
    return out


def fill(ranked_shells, k, what):
    """Берёт уровни целиком в порядке ранга, пока не наберётся ровно k МО."""
    chosen = []
    for sh in ranked_shells:
        if len(chosen) + len(sh) <= k:
            chosen += sh
        if len(chosen) == k:
            return chosen
    raise ValueError(f"{what}: нельзя набрать ровно {k} МО целыми уровнями")


def select_active(mf, mode, ne, no, o6_ids, k_ids):
    nocc = mf.mol.nelectron // 2
    e = mf.mo_energy
    if mode == "fermi":
        return list(range(nocc - ne // 2, nocc - ne // 2 + no))
    w_o6, w_k = atom_weights(mf, o6_ids), atom_weights(mf, k_ids)
    # виртуальные: самые низкие уровни с заметным весом на K (для ghost-K наибольший
    # вес у «ядерных» функций без ядра, ε ~ 10² Ha — поэтому ранг по энергии, а не весу)
    vir = fill(shells([i for i in range(nocc, len(e)) if w_k[i] > 0.3], e), no - ne // 2, "виртуальные")
    # занятые: уровни неподелённых пар O6, ранжированные по обменной связи с выбранными
    # K-орбиталями Σ_a (ia|ia) — одинаковое правило для TK и T(TK), баланс по симметрии
    cand = [i for i in range(nocc) if e[i] > -2.0 and w_o6[i] > 0.2]
    c = mf.mo_coeff
    eri = mf.with_df.ao2mo((c[:, cand], c[:, vir], c[:, cand], c[:, vir]), compact=False)
    eri = eri.reshape(len(cand), len(vir), len(cand), len(vir))
    score = {m: float(sum(eri[p, a, p, a] for a in range(len(vir)))) for p, m in enumerate(cand)}
    ranked = sorted(shells(cand, e), key=lambda sh: -np.mean([score[m] for m in sh]))
    occ = fill(ranked, ne // 2, "занятые")
    return occ + vir


def cas_qiskit(mf, act, ne, use_vqe, o6_ids, k_ids):
    no = len(act)
    mc = mcscf.DFCASCI(mf, no, ne)
    mo = mc.sort_mo([i + 1 for i in act])  # sort_mo — 1-based
    h1, ecore = mc.get_h1eff(mo)
    h2 = ao2mo.restore(1, mc.get_h2eff(mo), no)  # (ij|kl), химическая нотация
    ham = ElectronicEnergy.from_raw_integrals(h1, h2)
    ham.constants["core"] = ecore
    prob = ElectronicStructureProblem(ham)
    prob.num_particles = (ne // 2, ne // 2)
    prob.num_spatial_orbitals = no
    mapper = ParityMapper(num_particles=prob.num_particles)
    nq = mapper.map(ham.second_q_op()).num_qubits

    e_exact = GroundStateEigensolver(mapper, NumPyMinimumEigensolver()).solve(prob).total_energies[0]
    e_ref = mc.kernel(mo)[0]  # контроль: DF-CASCI PySCF
    assert abs(e_exact - e_ref) < 1e-7, (e_exact, e_ref)

    e_vqe, npar, t_vqe = None, None, 0.0
    if use_vqe:
        t0 = time.time()
        hf = HartreeFock(no, prob.num_particles, mapper)
        ans = UCCSD(no, prob.num_particles, mapper, initial_state=hf)
        npar = ans.num_parameters
        vqe = VQE(StatevectorEstimator(), ans, SLSQP(maxiter=300))
        vqe.initial_point = np.zeros(npar)
        e_vqe = GroundStateEigensolver(mapper, vqe).solve(prob).total_energies[0]
        t_vqe = time.time() - t0

    w_o6, w_k = atom_weights(mf, o6_ids), atom_weights(mf, k_ids)
    info = [(round(float(mf.mo_energy[i]), 3), round(float(w_o6[i]), 2), round(float(w_k[i]), 2)) for i in act]
    return dict(CAS=e_exact, VQE=e_vqe, nq=nq, npar=npar, t_vqe=t_vqe, act=info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="sto3g")
    ap.add_argument("--active", default="fermi", choices=["fermi", "o6k"])
    ap.add_argument("--ne", type=int, default=4)
    ap.add_argument("--no", type=int, default=4)
    ap.add_argument("--no-vqe", action="store_true")
    ap.add_argument("--chkdir", default="chk")
    args = ap.parse_args()
    vqe = not args.no_vqe
    os.makedirs(args.chkdir, exist_ok=True)

    syms, xyz, n = build_tetrad("auto")
    nt = 4 * n
    o6_ids, k_id = [k * n + O6 for k in range(4)], [nt]
    print(f"G-тетрада + K⁺: {len(syms)} атомов, базис {args.basis}, "
          f"CAS({args.ne}e,{args.no}o), активное пространство '{args.active}'\n", flush=True)

    calcs = {  # имя: (атомы, маска ghost, заряд)
        "TK": (len(syms), None, +1),
        "T(TK)": (len(syms), [False] * nt + [True], 0),
        "K(TK)": (len(syms), [True] * nt + [False], +1),
        "T": (nt, None, 0),
        "G(T)": (nt, [False] * n + [True] * (nt - n), 0),
    }
    E = {}
    for name, (na, mask, q) in calcs.items():
        t0 = time.time()
        tag = f"{args.basis}_{name.replace('(', '_').replace(')', '')}"
        chk = os.path.join(args.chkdir, tag + ".chk")
        res = os.path.join(args.chkdir, f"{tag}_{args.active}{args.ne}{args.no}{'' if vqe else '_novqe'}.json")
        if os.path.exists(res):  # результат уже посчитан (защита от потери долгого VQE)
            with open(res) as f:
                E[name] = json.load(f)
            print(f"[{name:6}] загружено из {res}", flush=True)
            continue
        mf = run_hf(atoms(syms[:na], xyz[:na], mask), args.basis, q, chk)
        e = {"HF": mf.e_tot, "MP2": mf.e_tot + DFMP2(mf).kernel(with_t2=False)[0], "nao": mf.mol.nao}
        line = f"[{name:6}] {e['nao']} АО, HF+MP2 {time.time() - t0:5.0f} с"
        if name == "K(TK)":
            e["CAS"] = e["VQE"] = mf.e_tot
        elif name in ("TK", "T(TK)"):
            act = select_active(mf, args.active, args.ne, args.no, o6_ids, k_id)
            e.update(cas_qiskit(mf, act, args.ne, vqe, o6_ids, k_id))
            line += f", {e['nq']} кубитов"
            if vqe:
                line += f", UCCSD {e['npar']} параметров, VQE {e['t_vqe']:.0f} с"
            line += "\n         активные МО (ε, вес O6, вес K): " + str(e["act"])
        E[name] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in e.items()}
        with open(res, "w") as f:
            json.dump(E[name], f, ensure_ascii=False, indent=1)
        print(line, flush=True)

    print("\nСвязывание K⁺ тетрадой, ΔE_K (ккал/моль, BSSE):")
    for key in ("HF", "MP2", "CAS", "VQE"):
        if E["TK"].get(key) is None:
            continue
        print(f"  {key:4} {(E['TK'][key] - E['T(TK)'][key] - E['K(TK)'][key]) * HA2KCAL:9.3f}")
    corr = (E["TK"]["CAS"] - E["TK"]["HF"]) - (E["T(TK)"]["CAS"] - E["T(TK)"]["HF"])
    print(f"  вклад корреляции CAS в ΔE_K = {corr * HA2KCAL:.3f} ккал/моль")
    if vqe:
        print(f"  |VQE−CAS| комплекс = {abs(E['TK']['VQE'] - E['TK']['CAS']) * HA2KCAL:.2e}, "
              f"тетрада = {abs(E['T(TK)']['VQE'] - E['T(TK)']['CAS']) * HA2KCAL:.2e} ккал/моль")

    print("\nОбразование тетрады из 4 G, ΔE_tetrad (ккал/моль, BSSE):")
    for key in ("HF", "MP2"):
        v = (E["T"][key] - 4 * E["G(T)"][key]) * HA2KCAL
        print(f"  {key:4} {v:9.2f}   (на H-связь {v / 8:6.2f})")


if __name__ == "__main__":
    main()
