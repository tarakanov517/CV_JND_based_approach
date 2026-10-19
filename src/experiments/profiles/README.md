# Профили аксонального шума

Эксперимент проверяет, как распределение аксонального шума между блоками V1, V2, V4 и IT влияет на clean accuracy и устойчивость к transfer-атакам. Шум применяется только во время обучения и выключается при оценке.

Всего используется 22 конфигурации и три seed: 42, 43 и 44. Получается 66 обучений по 50 эпох: 10 эпох классификатора и 40 эпох всей сети. Для каждой модели проверяются FGSM и PGD при ε=2/255, 4/255 и 8/255.

Сетка V1×V2 проверяет основные эффекты и взаимодействие двух наиболее перспективных блоков. Профили с одинаковой суммарной интенсивностью 1.6 сравнивают распределение шума при одинаковом бюджете. Возрастающие и убывающие профили показывают, важнее ли усиливать шум к поздним или к ранним блокам.

Запуск из корня проекта:

```bash
mkdir -p logs
TRAIN_JOB=$(sbatch --parsable experiments/profiles/run.sh | cut -d';' -f1)
TRANSFER_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB experiments/profiles/run_transfer.sh | cut -d';' -f1)
echo "TRAIN_JOB=$TRAIN_JOB TRANSFER_JOB=$TRANSFER_JOB"
```

После завершения transfer-задачи:

```bash
python3 experiments/profiles/summarize.py results/profiles
python3 experiments/profiles/plot.py results/profiles
```

В `results/profiles` появятся полная таблица, таблица средних и стандартных отклонений по трём seeds, сравнение профилей и тепловая карта взаимодействия V1×V2.
