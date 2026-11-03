# telomere-qiskit

Квантово-химическая модель фрагмента теломеры (`TTAGGG`)ₙ в `qiskit-nature`.

Берётся одно основание повтора (по умолчанию гуанин — основа G-квадруплексов
и главная мишень окислительного повреждения теломер) и считается в малом
активном пространстве вокруг HOMO/LUMO.

**Пайплайн:** SMILES → 3D (RDKit, MMFF) → HF/STO-3G (PySCF) → `ActiveSpaceTransformer`
→ `ParityMapper` → VQE(UCCSD) + точная диагонализация для проверки.

## Запуск

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python telomere_guanine.py                 # G, (4e,4o) -> 6 кубитов
.venv/bin/python telomere_guanine.py --base T        # A или T
.venv/bin/python telomere_guanine.py --ne 6 --no 6   # 10 кубитов
```

## Результат (G, STO-3G, 4e/4o, 6 кубитов)

| Метод | Энергия, Ha |
|---|---|
| Хартри–Фок | −532.44894 |
| VQE (UCCSD, 26 параметров) | −532.45789 |
| Точный (CASCI) | −532.45789 |

|VQE − exact| ≈ 1·10⁻⁶ Ha. Учебный масштаб: минимальный базис и малое активное
пространство — для отработки метода, не для сравнения с экспериментом.
