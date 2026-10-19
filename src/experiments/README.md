# Эксперименты

Все команды запускаются из корня проекта.

| Папка | Назначение |
|---|---|
| `base` | Основной эксперимент со всеми типами шумов |
| `curriculum` | Постепенное увеличение параметрического шума |
| `layerwise` | Послойное сравнение при фиксированной интенсивности |
| `intensity` | Зависимость метрик от интенсивности шума в отдельных блоках |
| `profiles` | Профили шума по глубине сети и асимметричная сетка V1×V2 |
| `adaptive` | Последовательный адаптивный рост шума с откатом по validation |
| `sweep` | Обработка старой широкой развертки интенсивностей |

Эксперимент `intensity` содержит 56 запусков на seed 42: одну baseline-модель и по пять интенсивностей `0.10`, `0.25`, `0.40`, `0.55`, `0.75` для аксонального шума. Проверяются блоки V1, V2, V4 и IT по отдельности, все шесть пар блоков и все блоки одновременно. В парной конфигурации оба блока получают одинаковую интенсивность. Каждая модель обучается 50 эпох: 10 эпох классификатор и 40 эпох вся сеть. Атаки проверяются при `epsilon = 2/255`, `4/255` и `8/255`.

Новый эксперимент запускается командой:

```bash
mkdir -p logs
TRAIN_JOB=$(sbatch --parsable experiments/intensity/run.sh | cut -d';' -f1)
TRANSFER_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB experiments/intensity/run_transfer.sh | cut -d';' -f1)
echo "TRAIN_JOB=$TRAIN_JOB TRANSFER_JOB=$TRANSFER_JOB"
```

После завершения transfer-задачи:

```bash
python3 experiments/intensity/summarize.py results/intensity
python3 experiments/intensity/plot.py results/intensity
```
