# Pipeline для проведения экспериментов и сравнения

## Модели

Практически во всех статьях рассматривается архитектура WideResNet с различными количеством слоев и коэффом ширины (70-16, 94-16, 82-8, 28-10). Есть ResNet (в основном ResNet-152). В последних статьях иногда тестируют Vit

## Датасеты

В качестве датасетов используются cifar 10, cifar 100, imagenet и его подверсия tiny-imagenet. Редко используется mnist

## Существующие методы

### 1. Состязательное обучение

1. [Towards Deep Learning Models Resistant to Adversarial
Attacks](https://arxiv.org/pdf/1706.06083)

2. [Theoretically Principled Trade-off between Robustness and Accuracy](https://arxiv.org/pdf/1901.08573)

3. [Boosting Adversarial Training Using Robust Selective Data Augmentation](https://link.springer.com/article/10.1007/s44196-023-00266-x)

### 2. Трансформация входов

1. [DiffPure: Diffusion Models for Adversarial Purification (Nie et al., ICML 2022)](https://arxiv.org/abs/2205.07460)

2. [DensePure: DensePure: Understanding Diffusion Models for Adversarial Robustness (Yuan et al., CVPR 2023)](https://arxiv.org/abs/2211.00322)

3. [Adversarial Purification via Super-Resolution and Diffusion (PuriFlow) (Park et al., ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Park_Adversarial_Purification_via_Super-Resolution_and_Diffusion_ICCV_2025_paper.pdf)

4. [MAE-based Purifier: Masked Autoencoders are Robust Data Purifiers (Liu et al., 2022/2023)](https://arxiv.org/html/2206.04846)

5. [Countering Adversarial Images using Input Transformations (ICLR 2018)](https://arxiv.org/abs/1711.00117)

    5 трансформаций изображения: 

    - (1) image cropping and rescaling, 
    - (2) bit-depth reduction, 
    - (3) JPEG compression, 
    - (4) total variance minimization
    - (5) image quilting.

## Атаки

1. [Robust Bench](https://github.com/RobustBench/robustbench)

    Бенчмарк для тестирования устойчивости к атакам + [лидерборд лучших методов защиты от атак](https://robustbench.github.io/)

    В нем применяется AutoAttack (Croce & Hein, 2020)

2. [ColorFool: Semantic Adversarial Colorization (CVPR 2020)](https://github.com/smartcameras/ColorFool)

    Атака на яркостные составляющие изображения (пространство Lab)

3. Классические атаки (FGSM, BIM, PGD)

4. BPDA

    Заменяем недифференцирумую операцию на ее аппроксимацию (самое простое g(x) = x) для выполнения градиентных атак

5. Square Attack (атака черного ящика). Входит в AutoAttack