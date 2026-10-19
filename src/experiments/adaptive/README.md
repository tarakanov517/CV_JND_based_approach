# Адаптивный последовательный рост шума

Каждый запуск начинается с baseline-чекпоинта соответствующего seed. Интенсивность аксонального шума последовательно увеличивается в блоках CORnet-RT. Проверяются три порядка: V1→V2→V4→IT, обратный порядок и V2→V1→V4→IT.

В каждом блоке проверяются уровни 0.25, 0.40, 0.55, 0.70 и 0.85. На каждой ступени модель дообучается четыре эпохи. Уровень сохраняется, если transfer PGD на фиксированной validation-выборке улучшается и clean validation accuracy остаётся не ниже baseline минус 10 процентных пунктов. После выбора всех уровней выполняются 10 финальных эпох с зафиксированным профилем.

Поиск выполняется на seeds 42, 43 и 44. Всего запускается девять адаптивных обучений. Тестовая выборка не участвует в выборе уровней и используется только для финальной оценки.

Эксперимент использует baseline-модели из `results/profiles`, поэтому сначала должен полностью завершиться эксперимент `profiles`.

Запуск из корня проекта:

```bash
mkdir -p logs
ADAPTIVE_JOB=$(sbatch --parsable experiments/adaptive/run.sh | cut -d';' -f1)
TRANSFER_JOB=$(sbatch --parsable --dependency=afterok:$ADAPTIVE_JOB experiments/adaptive/run_transfer.sh | cut -d';' -f1)
echo "ADAPTIVE_JOB=$ADAPTIVE_JOB TRANSFER_JOB=$TRANSFER_JOB"
```

После завершения:

```bash
python3 experiments/adaptive/summarize.py results/adaptive
python3 experiments/adaptive/plot.py results/adaptive
```
