# Расширенный адаптивный эксперимент

Эксперимент заново начинается с baseline-модели и сравнивает два порядка последовательного роста аксонального шума:

- `forward`: V1 → V2 → V4 → IT;
- `reverse`: IT → V4 → V2 → V1.

Для каждого блока проверяются уровни `0.10`, `0.25`, `0.40`, `0.55`, `0.70`, `0.85`, `1.00`, `1.25`, `1.50`, `1.75`, `2.00`, `2.25` и `2.50`. На каждом уровне выполняется четыре эпохи. Если clean validation accuracy падает более чем на 10 процентных пунктов относительно baseline, состояние модели откатывается, а поиск продолжается в следующем блоке.

Если все 13 уровней принимаются во всех четырёх блоках, поиск занимает 208 эпох. Затем выполняются 42 эпохи консолидации, поэтому максимальная продолжительность составляет 250 эпох. При раннем отклонении уровней запуск завершится раньше.

Проверяются seed 42, 43 и 44. Всего обучается шесть моделей.

Запуск из корня проекта:

```bash
mkdir -p logs
ADAPTIVE_JOB=$(sbatch --parsable experiments/adaptive_extended/run.sh | cut -d';' -f1)
TRANSFER_JOB=$(sbatch --parsable --dependency=afterok:$ADAPTIVE_JOB experiments/adaptive_extended/run_transfer.sh | cut -d';' -f1)
echo "ADAPTIVE_JOB=$ADAPTIVE_JOB TRANSFER_JOB=$TRANSFER_JOB"
```

После завершения transfer-задачи:

```bash
python3 experiments/adaptive/summarize.py results/adaptive_extended
python3 experiments/adaptive/plot.py results/adaptive_extended
```

Результаты сохраняются в `results/adaptive_extended`, поэтому данные предыдущего эксперимента в `results/adaptive` не изменяются.
