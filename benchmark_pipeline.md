# Pipeline для проведения экспериментов и сравнения

## Модели

Практически во всех статьях рассматривается архитектура WideResNet с различными количеством слоев и коэффом ширины (70-16, 94-16, 82-8, 28-10). Есть ResNet (в основном ResNet-152). В последних статьях иногда тестируют Vit

## Датасеты

В качестве датасетов используются cifar 10, cifar 100, imagenet и его подверсия tiny-imagenet. Редко используется mnist

## Существующие методы

### 1. Состязательное обучение

1. [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/pdf/1706.06083)

    Обучение классификатора на атакованных примерах, полученных с помощью атаки PGD

2. [Theoretically Principled Trade-off between Robustness and Accuracy](https://arxiv.org/pdf/1901.08573)

    Учим классификатор со специальным взвешенным лоссом, который состоит из 2 частей:

    1. Кросс-энтропия на чистых данных, хотим высокую точности на чистых данных
    2. KL-дивергенция (расхождение Кульбака — Лейблера), штрафуем за расхождение распределения выдаваемых вероятностей на чистом и атакованном изображениях

3. [Boosting Adversarial Training Using Robust Selective Data Augmentation](https://link.springer.com/article/10.1007/s44196-023-00266-x)

    То, что кидала Лиза. Контроль переобучения при состязательном обучении за счет применения сильных аугментаций только к сложным изображениям (которые модель часто путает) 

### 2. Трансформация входов

1. [DiffPure: Diffusion Models for Adversarial Purification (Nie et al., ICML 2022)](https://arxiv.org/abs/2205.07460)

    Очистка картинок через диффузионные модели. К атакованному изображению добавляется небольшой прямой диффузионный шум для разрушения атаки, после чего обратный процесс восстановления создает чистую картинку

2. [DensePure: DensePure: Understanding Diffusion Models for Adversarial Robustness (Yuan et al., CVPR 2023)](https://arxiv.org/abs/2211.00322)

    Улучшение DiffPure против адаптивных атак за счет борьбы со случайностью обратного-процесса. Метод создает множество различных траекторий восстановления из одного зашумленного состояния и усредняет их

3. [Adversarial Purification via Super-Resolution and Diffusion (PuriFlow) (Park et al., ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Park_Adversarial_Purification_via_Super-Resolution_and_Diffusion_ICCV_2025_paper.pdf)

    Двухстадийная очистка. Модель Super-Resolution удаляет высокочастотный шум и сжимает пространство, а затем легкая модель диффузии дорисовывает детали текстур

4. [MAE-based Purifier: Masked Autoencoders are Robust Data Purifiers (Liu et al., 2022/2023)](https://arxiv.org/html/2206.04846)

    Используется Masked Autoencoders (MAE) на ViT. Картинка случайно маскируется на 70–80% (уничтожая большую часть состязательного шума), а автоэнкодер дорисовывает недостающие блоки, опираясь только на оставшиеся патчи

5. [Countering Adversarial Images using Input Transformations (ICLR 2018)](https://arxiv.org/abs/1711.00117)

    5 трансформаций изображения: 

    - (1) image cropping and rescaling - кадрирование и масштабирование изображения
    - (2) bit-depth reduction - уменьшение битовой глубины - квантование цвета
    - (3) JPEG compression - сжатие в jpeg
    - (4) total variance minimization - оптимизационная задача поиска изображения, близкое к начальному, но с минимальной суммой абсолютных градиентов пикселей:
    $$
        \hat{I} = \arg\min_{\hat{I}} \left( \underbrace{\|\hat{I} - I_{\text{orig}}\|^2}_{\text{штраф за отклонение от оригинала}} + \lambda \underbrace{\sum_{i,j} \left( |\nabla_x \hat{I}(i,j)| + |\nabla_y \hat{I}(i,j)| \right)}_{\text{регуляризатор полной вариации}} \right)
    $$

    $$
        \nabla_x I(i,j) = I(i+1, j) - I(i, j), \quad \nabla_y I(i,j) = I(i, j+1) - I(i, j)
    $$

    $$
        \nabla I(i,j) = \begin{pmatrix} \nabla_x I(i,j) \\ \nabla_y I(i,j) \end{pmatrix} = \begin{pmatrix} I(i+1, j) - I(i, j) \\ I(i, j+1) - I(i, j) \end{pmatrix}
    $$
    - (5) image quilting - разбиваем изображение на патчи с перекрытиями, затем заменяем каждый патч на самый близкий патч из заранее собранного датасета чистых патчей, дальше сшиваются края

6.  [Adversarial Perturbations Prevail in the Y-Channel of the YCbCr Color Space (25 Feb 2020)](https://arxiv.org/abs/2003.00883)

    Метод преобразования изображения в YCbCr, далее к Y каналу яркости добавляется случайный шум, затем этот канал пропускается через ResUpNet сетку с опорой через skip connections на карты признаков, полученных при подаче исходной RGB картинки в ResNet18. После очищенный Y канал объединяется с Cb и Cr и преобразуется  обратно в RGB

## Атаки

1. [Robust Bench](https://github.com/RobustBench/robustbench)

    Бенчмарк для тестирования устойчивости к атакам + [лидерборд лучших методов защиты от атак](https://robustbench.github.io/)

    В нем применяется набор атак AutoAttack. Состоит из APGD-CE, APGD-DLR, FAB и Square Attack

2. [ColorFool: Semantic Adversarial Colorization (CVPR 2020)](https://github.com/smartcameras/ColorFool)

    Особо не применяют, но интересно и полезно будет потестить, так как прямо к нам относится. Атака на яркостные составляющие изображения (пространство Lab)

3. Классические атаки (FGSM, BIM, PGD)

4. BPDA (необходима, так как напрямую не можем атаковать градиентными методами ([статейка про обфускацию градиентов](https://arxiv.org/abs/1802.00420)))

    Заменяем недифференцирумую операцию на ее аппроксимацию (самое простое g(x) = x) для выполнения градиентных атак

5. Square Attack (атака черного ящика). Входит в AutoAttack

6. EOT (скорее для товарищей, которые работают с добавлением шума = случайности и возможно будут шумить еще и на этапе инференса (но это не точно))

    Грубо говоря усреднение градиента по нескольким проходам