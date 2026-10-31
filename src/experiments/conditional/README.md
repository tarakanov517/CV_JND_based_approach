# Условная модель аксонального шума

Один checkpoint обучается работать с разными профилями аксонального шума. Профиль содержит четыре значения в порядке V1, V2, V4 и IT. Он одновременно управляет интенсивностью параметрического шума и поступает в conditioner, который создаёт отдельные масштаб и сдвиг признаков для каждого блока.

На каждом обучающем батче выбирается один профиль. Используется смесь четырёх режимов:

- 20% батчей без шума;
- 30% с одинаковой случайной интенсивностью во всех блоках;
- 30% с шумом в одном случайном блоке;
- 20% с четырьмя независимыми интенсивностями.

Первые пять эпох проходят без шума, затем максимальная интенсивность за 25 эпох линейно увеличивается до 1.0. Всего выполняется 100 эпох. Обучаются три модели для seed 42, 43 и 44.

После обучения одна и та же модель проверяется при профилях `zero`, `all_025`, `all_050`, `all_085`, `all_100`, `v1_050`, `v2_050`, `v4_050` и `it_050`. Clean и baseline-transfer метрики считаются на полном тесте с усреднением трёх стохастических прогонов. White-box атаки считаются на 512 изображениях. Для обхода случайности используется EOT с четырьмя реализациями шума на каждом шаге атаки.

Baseline-модели должны находиться в `results/profiles/baseline/seed_<seed>`.

Запуск из корня проекта:

```bash
mkdir -p logs
TRAIN_JOB=$(sbatch --parsable experiments/conditional/run.sh | cut -d';' -f1)
EVAL_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB experiments/conditional/run_eval.sh | cut -d';' -f1)
echo "TRAIN_JOB=$TRAIN_JOB EVAL_JOB=$EVAL_JOB"
```

После завершения:

```bash
source "$HOME/venvs/cornet/bin/activate"
python3 experiments/conditional/summarize.py results/conditional_yoto
python3 experiments/conditional/plot.py results/conditional_yoto
```

Итоговые файлы:

- `conditional_results.csv` — все профили и seed;
- `conditional_summary.csv` — средние значения и стандартные отклонения;
- `conditional_evaluation.png` — сравнение clean, transfer и EOT white-box метрик;
- `conditional_training.png` — качество при нулевом и опорном профиле во время обучения.
