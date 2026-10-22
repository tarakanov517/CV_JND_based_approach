# CV_JND_based_approach

Перцептивно-обоснованная робастность классификаторов: обучаемый фронт-энд из психофизики
контрастных порогов (Симкин) + нейронный шум, в сравнении с биологическим VOneNet.

## Структура

- `src/` — модели и обучение:
  - `tile_model.py` — `TileResNet` (голова Симкина на `tap`, шум на `noise_tap`, additive/neuronal);
  - `model.py` — сиамский `CustomResNet` (два поля) + `DualBN`;
  - `vone_model.py` — `VOneResNet50` (обёртка официального VOneNet, `src/VOne/` — вендорённый код);
  - `train_tile.py`, `train_late_simkin.py`, `train_vone.py`, `train_resnet.py` — пайплайны обучения;
  - датасеты: `tile_dataset.py`, `custom_datasets.py`, `cls_data.py`;
  - стимулы Симкина: `stimulus.py`, `display.py`, `visibility_funcs.py`; `corruptions.py` — ImageNet-C.
- `scripts/` — генерация синтетики (`generate_tile.py`, `generate_8_1.py`, `generate_7_3.py`)
  и оценка (`eval_blackbox.py`, `bench.py`, `noise_ablation.py`, `eval_sigma_eff.py`).
- `configs/` — конфиги бенчмарка (`bench_imagenette.yaml`, `bench_stl10.yaml`, `eval_sigma_eff.yaml`).
- `params.yaml` — единый конфиг обучения/генерации (секции `general`, `display`, `stimulus`,
  `train_tile`, `train_siamese`, `late_simkin`, `noise_ablation`, `exp224`).

## Установка

```
pip install -e .        # зависимости из pyproject.toml, Python >= 3.12
```

## Пайплайны

Все скрипты запускаются из корня репозитория; CLI-оверрайды — через OmegaConf (`key=value`).

**1. Синтетика (стимулы Симкина)** -> в `general.output_folder` из `params.yaml`:
```
python scripts/generate_tile.py        # одиночные тайлы (для late-simkin)
python scripts/generate_8_1.py         # серия 8.1 [A19]
python scripts/generate_7_3.py         # серия 7.3
```

**2. STL-10 (96px):**
```
python src/train_tile.py                           # tile-модель
python src/train_late_simkin.py                    # late-simkin (голова l4, шум conv1)
python scripts/noise_ablation.py                   # σ-свип late-simkin vs baseline
```

**3. Нативные 224 (imagenette / imagewoof):**
```
python src/train_late_simkin.py preset=exp224                 # baseline + late-simkin
python src/train_vone.py noise_scales=[0.70,0.35,0.20,0.10]   # VOne-свип по дозе шума
```

**4. Оценка:**
```
python scripts/bench.py configs/bench_imagenette.yaml         # clean / whitebox / eot / blackbox / corruptions
python scripts/eval_sigma_eff.py configs/eval_sigma_eff.yaml  # Simkin vs VOne на общей оси σ_eff
```