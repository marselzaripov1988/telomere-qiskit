"""
Классический эталон: потенциал ионизации гуанина (G -> G⁺).

Геометрии: 9H-кето-амино гуанин (таутомер ДНК), B3LYP/cc-pVDZ,
  нейтраль (RKS) и катион (UKS, дублет) — оптимизация через geomeTRIC.
Вертикальный IP  (при геометрии нейтрали): E(G⁺) − E(G)
Адиабатический IP (каждый при своей геометрии).

Методы (frozen core где применимо):
  ΔHF, ΔMP2, ΔCCSD, ΔCCSD(T)        — cc-pVDZ (UHF-референс для катиона)
  EOM-IP-CCSD                        — cc-pVDZ, вертикальный
  CASSCF / SC-NEVPT2                 — всё π-пространство (AVAS по 2pz C/N/O)
  поправка на базис: ΔMP2/aug-cc-pVDZ − ΔMP2/cc-pVDZ, добавляется к CCSD(T)

Эксперимент: VIE 8.24 эВ (PES, Hush & Cheung), AIE 7.77 эВ (Orlov et al.).
Результаты каждого шага кэшируются в results/g_cation.json.
"""
import json
import os
import time

import numpy as np
from pyscf import gto, scf, mp, cc, mcscf, mrpt, dft
from pyscf.mcscf import avas
from pyscf.geomopt.geometric_solver import optimize
from rdkit import Chem
from rdkit.Chem import AllChem

HA2EV = 27.211386
EXP = {"vertical": 8.24, "adiabatic": 7.77}
OUT = "results/g_cation.json"
SMILES_G = "Nc1nc2[nH]cnc2c(=O)[nH]1"
MEM = 18000
CC_MEM = 9000  # МБ на CC; остальное — запас под (T) и ОС (всего 23 ГБ)


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save(R):
    os.makedirs("results", exist_ok=True)
    json.dump(R, open(OUT, "w"), indent=1, ensure_ascii=False)


def cached(R, key, fn):
    if key not in R:
        t0 = time.time()
        R[key] = fn()
        R[key + "_t"] = round(time.time() - t0, 1)
        save(R)
        print(f"  {key:28s} = {R[key]}   ({R[key + '_t']} с)", flush=True)
    return R[key]


def planarize(xyz):
    """Центр в 0, плоскость колец -> xy (нужно для AVAS по 2pz)."""
    xyz = np.asarray(xyz) - np.mean(xyz, axis=0)
    _, _, vt = np.linalg.svd(xyz)
    if np.linalg.det(vt) < 0:
        vt[2] *= -1
    return xyz @ vt.T


def rdkit_start():
    mol = Chem.AddHs(Chem.MolFromSmiles(SMILES_G))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    return [a.GetSymbol() for a in mol.GetAtoms()], mol.GetConformer().GetPositions()


def mole(sym, xyz, basis, charge):
    return gto.M(atom=list(zip(sym, map(tuple, xyz))), basis=basis, charge=charge,
                 spin=charge % 2, verbose=0, max_memory=MEM)


def opt_geometry(sym, xyz, charge):
    m = mole(sym, xyz, "cc-pvdz", charge)
    mf = (dft.RKS(m) if charge == 0 else dft.UKS(m)).density_fit()
    mf.xc = "b3lyp"
    m_eq = optimize(mf, maxsteps=200)
    return planarize(m_eq.atom_coords(unit="Angstrom")).tolist()


def scf_run(m):
    mf = scf.RHF(m) if m.spin == 0 else scf.UHF(m)
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        mf = scf.newton(mf).run()
    assert mf.converged
    return mf


def s2(mf):
    return float(mf.spin_square()[0]) if m_is_open(mf) else 0.0


def m_is_open(mf):
    return mf.mol.spin != 0


def corr_energies(sym, xyz, charge):
    m = mole(sym, xyz, "cc-pvdz", charge)
    mf = scf_run(m)
    ncore = 11  # 1s C(5) N(5) O(1)
    e_mp2 = mp.MP2(mf, frozen=ncore).run().e_tot
    mycc = cc.CCSD(mf, frozen=ncore)
    mycc.conv_tol = 1e-8
    mycc.max_memory = CC_MEM
    mycc.direct = True  # vvvv через AO-интегралы, без хранения (иначе OOM у UCCSD)
    mycc.kernel()
    e_t = mycc.ccsd_t()
    # T1-диагностика = ||t1||(спин-орбитали) / sqrt(N коррелируемых электронов)
    if isinstance(mycc.t1, tuple):
        t1n2 = sum(float(np.sum(t ** 2)) for t in mycc.t1)
    else:
        t1n2 = 2 * float(np.sum(mycc.t1 ** 2))
    return {"HF": mf.e_tot, "MP2": e_mp2, "CCSD": mycc.e_tot, "CCSD(T)": mycc.e_tot + e_t,
            "S2": s2(mf), "t1diag": float(np.sqrt(t1n2 / (m.nelectron - 2 * ncore)))}


def mp2_aug(sym, xyz, charge):
    mf = scf_run(mole(sym, xyz, "aug-cc-pvdz", charge))
    return {"HF": mf.e_tot, "MP2": mp.MP2(mf, frozen=11).run().e_tot}


def eom_ip(sym, xyz):
    mf = scf_run(mole(sym, xyz, "cc-pvdz", 0))
    mycc = cc.RCCSD(mf, frozen=11)
    mycc.max_memory = CC_MEM
    mycc.direct = True
    mycc.kernel()
    e, _ = mycc.ipccsd(nroots=3)
    return [float(x) * HA2EV for x in np.atleast_1d(e)]


def cas_nevpt2(sym, xyz_neutral_ref, xyz, charge):
    """π-CAS: орбитали AVAS(2pz) из RHF нейтрали в той же геометрии; катион — то же
    пространство с одним электроном меньше (ROHF-дублет, CASSCF)."""
    m0 = mole(sym, xyz, "cc-pvdz", 0)
    mf0 = scf_run(m0)
    ncas, nelecas, mo = avas.avas(mf0, ["C 2pz", "N 2pz", "O 2pz"], canonicalize=False)
    if charge == 0:
        mc = mcscf.CASSCF(mf0.density_fit(), ncas, nelecas)
    else:
        mfc = scf.ROHF(mole(sym, xyz, "cc-pvdz", charge)).run(conv_tol=1e-10)
        ne = nelecas - 1
        mc = mcscf.CASSCF(mfc.density_fit(), ncas, ((ne + 1) // 2, ne // 2))
    # неплоская (пирамидальная NH2) нейтраль: медленное σ/π-смешивание орбиталей
    mc.max_cycle_macro = 300
    mc.conv_tol = 1e-7
    mc.conv_tol_grad = 1e-3
    mc.fcisolver.conv_tol = 1e-10
    mc.kernel(mo)
    assert mc.converged, "CASSCF не сошёлся"
    e_pt2 = mrpt.NEVPT(mc).kernel()
    return {"CAS": f"({mc.nelecas[0] + mc.nelecas[1]}e,{ncas}o)", "CASSCF": mc.e_tot,
            "NEVPT2": mc.e_tot + float(e_pt2)}


def main():
    R = load()
    sym, x0 = rdkit_start()
    print("1) Геометрии B3LYP/cc-pVDZ", flush=True)
    g0 = cached(R, "geom_neutral", lambda: opt_geometry(sym, x0, 0))
    gc = cached(R, "geom_cation", lambda: opt_geometry(sym, g0, 1))

    print("2) Корреляционные методы, cc-pVDZ", flush=True)
    n_n = cached(R, "neutral@n", lambda: corr_energies(sym, g0, 0))
    c_n = cached(R, "cation@n", lambda: corr_energies(sym, g0, 1))
    c_c = cached(R, "cation@c", lambda: corr_energies(sym, gc, 1))

    print("3) Поправка на базис aug-cc-pVDZ (MP2)", flush=True)
    a_nn = cached(R, "aug neutral@n", lambda: mp2_aug(sym, g0, 0))
    a_cn = cached(R, "aug cation@n", lambda: mp2_aug(sym, g0, 1))
    a_cc = cached(R, "aug cation@c", lambda: mp2_aug(sym, gc, 1))

    print("4) EOM-IP-CCSD/cc-pVDZ", flush=True)
    eom = cached(R, "eom_ip_eV", lambda: eom_ip(sym, g0))

    print("5) π-CASSCF / NEVPT2, cc-pVDZ", flush=True)
    cas_nn = cached(R, "cas neutral@n", lambda: cas_nevpt2(sym, g0, g0, 0))
    cas_cn = cached(R, "cas cation@n", lambda: cas_nevpt2(sym, g0, g0, 1))
    cas_cc = cached(R, "cas cation@c", lambda: cas_nevpt2(sym, g0, gc, 1))

    ev = lambda a, b: (a - b) * HA2EV
    rows = []
    for meth in ("HF", "MP2", "CCSD", "CCSD(T)"):
        rows.append((f"Δ{meth}/cc-pVDZ", ev(c_n[meth], n_n[meth]), ev(c_c[meth], n_n[meth])))
    corr_v = ev(a_cn["MP2"], a_nn["MP2"]) - ev(c_n["MP2"], n_n["MP2"])
    corr_a = ev(a_cc["MP2"], a_nn["MP2"]) - ev(c_c["MP2"], n_n["MP2"])
    rows.append(("ΔMP2/aug-cc-pVDZ", ev(a_cn["MP2"], a_nn["MP2"]), ev(a_cc["MP2"], a_nn["MP2"])))
    rows.append(("ΔCCSD(T)/cc-pVDZ + δaug(MP2)", rows[3][1] + corr_v, rows[3][2] + corr_a))
    rows.append(("EOM-IP-CCSD/cc-pVDZ", eom[0], None))
    rows.append((f"CASSCF{cas_nn['CAS']}/cc-pVDZ", ev(cas_cn["CASSCF"], cas_nn["CASSCF"]),
                 ev(cas_cc["CASSCF"], cas_nn["CASSCF"])))
    rows.append(("SC-NEVPT2/cc-pVDZ", ev(cas_cn["NEVPT2"], cas_nn["NEVPT2"]),
                 ev(cas_cc["NEVPT2"], cas_nn["NEVPT2"])))

    print(f"\n{'Метод':34s} {'VIE, эВ':>8} {'AIE, эВ':>8}")
    for name, v, a in rows:
        print(f"{name:34s} {v:8.3f} {a:8.3f}" if a is not None else f"{name:34s} {v:8.3f} {'—':>8}")
    print(f"{'Эксперимент':34s} {EXP['vertical']:8.2f} {EXP['adiabatic']:8.2f}")
    print(f"\n<S²> UHF катиона: {c_n['S2']:.3f} (верт.), {c_c['S2']:.3f} (адиаб.); идеал 0.75")
    print(f"T1-диагностика: G {n_n['t1diag']:.3f}, G⁺ {c_n['t1diag']:.3f}")
    print(f"EOM-IP-CCSD, первые 3 состояния G⁺: {np.round(eom, 3).tolist()} эВ")
    R["table"] = rows
    save(R)


if __name__ == "__main__":
    main()
