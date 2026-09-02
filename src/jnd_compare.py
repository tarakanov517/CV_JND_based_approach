import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import rootutils
import cv2
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from torch import optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from datasets import load_dataset
from tqdm import tqdm
import torchattacks

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.custom_datasets import STL10RGBDataset
from scripts.eval_blackbox import load_model, get_loader01, eot_pgd_acc


L_MIN, L_MAX = 0.1, 300.0
CACHE_DIR = "jnd_cache"
JND_CKPT = "models/ResNet50_jnd_kkk_stl10.pth"
JND_EPOCHS = 60
JND_LR = 1e-3
N_EVAL = 1000
EPS_LIST = [2 / 255, 4 / 255, 8 / 255, 16 / 255]
PGD_STEPS = 20
EOT_K = 20
JND_DIST = True                                     # Lp и в JND-домене (дорого: лишний трансформ на батч)
RGB_CKPT = "models/ResNet50_rgb_stl10.pth"

MY_CKPT = "models/ResNetLateSimkin_l4_best.pt"
MY_KW = dict(arch="tile", tap="l4", noise_tap="conv1", dual_bn=True)
MY_NOISE = 0.5


def make_backbone():
    m = models.resnet50(weights="IMAGENET1K_V1")
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m


class JNDModel:
    
    def __init__(self, L_MIN = 0.1, L_MAX = 300, t = 1, phi_d_screen = 1e-3, p = 0.75):
        self.L_MIN = L_MIN
        self.L_MAX = L_MAX
        self.t = t # время экспозиции (наблюдения) объекта в секундах
        self.phi_d_screen = phi_d_screen # угловой размер пикселя на экране (в радианах)
        self.phi_d = phi_d_screen # угловой размер объекта (в радианах)
        self.p = p # требуемая вероятность обнаружения объекта
        self.L0 = 1e-6
    
    @classmethod
    def from_screen_params(cls, screen_diag_inch = 16, screen_width_px = 1920, screen_height_px = 1080, distance_m = 0.5,
                           l_gray = 60, **kwargs):
        '''
        Фабричный метод для инициализации JNDModel с рассчетом phi_d на основе параметров экрана
        '''
        diag_px = math.sqrt(screen_width_px ** 2 + screen_height_px ** 2)
        ppi = diag_px / screen_diag_inch
        
        pixel_size_m = 0.0254 / ppi # 1 дюйм = 0.0254 метра

        phi_d_screen = pixel_size_m / distance_m # считаем угловой размер пикселя на экране

        instance = cls(phi_d_screen=phi_d_screen, **kwargs)

        instance.screen_width_px = screen_width_px
        instance.screen_height_px = screen_height_px
        instance.ppi = ppi
        instance.l_gray = l_gray

        return instance

    def _K(self, lambd, b1 = 0.98, b2 = 0.1, lambd1 = 2e-2):
        lambd = np.asarray(lambd, dtype = np.float64)
        return np.where(lambd <= 1, b1 * (1 + lambd1 / lambd), lambd ** b2)
    
    def _A(self, La, a1 = 4.8e6, a2 = 7.1e3, a3 = 4.5e-4): # единицы измерения параметров a1, a2, a3: м ** 2 / кд
        '''
        Функция, характеризующая ослабление сигнала в зрительной системе в результате процесса яркостной адаптации 
        к однородному фону с яркостью La

        '''
        return 1 + (a1 * La) ** 0.5 + a2 * La * (1 + (a3 * La) ** 0.5) # формула 9.6
    
    def _Lambda(self, lambd):
        return lambd * self._K(lambd)
    
    def _eta(self, La, L3 = 1e6, L4 = 1, a4 = 600): # единицы измерения параметров L3: кд / м ** 2, L4: кд / м ** 2, a4: м ** 4 / кд ** 2
        '''
        Функция адаптации временных интервалов

        '''
        return 1 + L3 / (1 + a4 * La * (L4 + La)) # Формула 9.12
    
    def _C(self, La, t): # единицы измерения: c
        tao_min = 0.05 # минимальное значение времени инерции зрения
        T = t / (tao_min * self._eta(La)) # формула 9.7
        return 1 + 1 / T # Формула 9.10. При имплементации выбрал 1 вариант из статьи. Есть 2 вариант - формула 9.11: C = 1 / (1 - exp(-T))
    
    def _func_phi_1(self, La, c2 = 0.25, L1 = 10, L2 = 2.6e-5): # единицы измерения параметров L1, L2: кд / м ** 2
        return (1 + L1 / (L2 + La)) ** c2 # формула 9.4
    
    def _S(self, La, theta_1 = 6.1, a_d = 2, inf_phi_1 = 1):
        phi_0_min = 0.5 * (1 / 60) * (np.pi / 180) # phi_0_min = 0,5' - 1'
        phi_1 = self._func_phi_1(La)
        phi_0 = phi_1 * phi_0_min / inf_phi_1
        
        return (theta_1 * phi_0 / self.phi_d + 1.0) ** a_d # формула 9.2
    
    def _l(self, La):
        return self.L0 * self._A(La) # формула 9.5
    
    def _P(self, p, p_ref = 0.75):
        '''
        Пороговая функция вероятности обнаружения, в статье не описано что это за функция. 
        Решил взять отношение необходимых сдвигов распределения при которых площадь по графиком шума глаза + сигнала равна соотвественно p_ref и p

        p_ref - веротность, с котороый мы РЕАЛЬНО в экспериментах статьи видим разницу в измении на 1 уровень яркости
        p - веротность, с котороый мы хотим чтобы была видна разница в измении на 1 уровень яркости

        В статье p_ref = 0.75 не фигурирует, так что в экспериментах используется p = p_ref = 0.75, чтобы множитель 
        от функции P не вносил ничего в итоговую формулу (P = 1)

        '''
        return abs(norm.ppf(p) / norm.ppf(p_ref))

    def _D(self, L, La): # единицы измерения параметра L0: кд / м ** 2
        '''
        L0 - абсолютный световой порог - минимальное значение яркости, которое человек способен заметить

        A(La) - адаптационное ослабление сигнала, зависит от яркости адаптации, регулировка чувствительности 
        к яркости в зависимости от фона

        C(La, t) - временная обработка, зависит от времени экспозиции t и яркости адаптации

        Lambda(lambd) - амплитудная обработка, зависит от lambd - нормализованной разницой между яркостью объекта 
        и яркостью адаптации, описывает нелинейность вопсприятия

        S(La, phi_d) - пространственная обработка, учитывает угловой размер объекта (phi_d) и минимальный угол разрешения глаза (phi_0)

        P(p) - пороговая функция вероятности обнаружения, позволяет учесть вероятность с которой мы хотим видеть объект

        '''
        lambd = L / (10 ** 2 * self._l(La)) # формула 9.13
        return self.L0 * self._A(La) * self._C(La, self.t) * self._Lambda(lambd) * self._S(La) * self._P(self.p) # формула 9.16

    def find_La(self, L_matrix):
        EPS = 1e-12
        logL = np.log(np.maximum(L_matrix, EPS))
    
        # Границы можно расширить, если адаптация может лежать вне диапазона яркостей картинки.
        log_La_min = float(logL.min())
        log_La_max = float(logL.max())
    
        def S_score(L_matrix, start_La):
            numerator = self._D(L_matrix, start_La)
            denominator = self._D(L_matrix, L_matrix)
            return float(np.mean(np.log(numerator / denominator)))
        
        def objective_log_La(log_La):
            return S_score(L_matrix, math.exp(float(log_La)))
        
        result = minimize_scalar(
            objective_log_La,
            bounds=(log_La_min, log_La_max),
            method='bounded',
            options={'xatol': 1e-6},
        )
        
        log_La_star = float(result.x)
        self.La = math.exp(log_La_star)
        
        return self.La


    def build_level_boundaries(self):
        L_right_bounds = []
        L_curr = self.La
        while L_curr < self.L_MAX:
            L_curr += self._D(L_curr, self.La)
            L_right_bounds.append(min(L_curr, self.L_MAX))

        L_left_bounds = []
        L_curr = self.La
        while L_curr > self.L0:
            L_curr -= self._D(L_curr, self.La)
            L_left_bounds.append(max(L_curr, self.L0))

        self.len_left = len(L_left_bounds)

        L_left_bounds.reverse()

        self.bounds = np.array(L_left_bounds + [self.La] + L_right_bounds)

        return self.bounds
    
    def L_to_k(self, L): # всегда берем нижнюю границу интервала, в который попали
        return np.searchsorted(self.bounds, L, side = 'right') - 1 - self.len_left

class ImageConverter:
    def __init__(self):
        self.M = np.array([
            [0.4124564,  0.3575761,  0.1804375],
            [0.2126729,  0.7151522,  0.0721750],
            [0.0193339,  0.1191920,  0.9503041]
        ])
        self.M_inv = np.array([
            [ 3.1338561, -1.6168667, -0.4906146],
            [-0.9787684,  1.9161415,  0.0334540],
            [ 0.0719453, -0.2289914,  1.4052427]
        ])

    def read_img(self, image):
        V = cv2.imread(image)
        V = cv2.cvtColor(V, cv2.COLOR_BGR2RGB)
        return V.astype(np.float64)

    def normalize_image(self, image):
        if image.max() > 1:
            image = image / 255.0
        return image

    def sRGB_Companding(self, linear_image):
        linear = np.clip(np.asarray(linear_image, dtype=np.float64), 0.0, 1.0)
        
        encoded = np.where(
            linear <= 0.0031308,
            linear * 12.92,
            1.055 * (linear ** (1.0 / 2.4)) - 0.055
        )
        return encoded

    def Inverse_sRGB_Companding(self, image):
        img = np.asarray(image, dtype=np.float64)
        
        linear = np.where(
            img <= 0.04045,
            img / 12.92,
            ((img + 0.055) / 1.055) ** 2.4
        )
        return linear

    def Linear_RGB_to_XYZ(self, image):
        XYZ_image = image.reshape(-1, 3)
        return (XYZ_image @ self.M.T).reshape(image.shape)

    def XYZ_to_xyY(self, XYZ_image, L_MAX=300.0, coord_white=(0.3127, 0.3290)):
        sum_xyz = XYZ_image.sum(axis=-1, keepdims=True)
        eps = 1e-10
        xy = XYZ_image[..., :2] / (sum_xyz + eps)
        is_black = (sum_xyz.squeeze(-1) < eps)
        if np.any(is_black):
            xy[is_black] = coord_white
        Y = XYZ_image[..., 1]
        L_norm = np.clip(Y, 0.0, 1.0)
        return np.stack([xy[..., 0], xy[..., 1], L_norm], axis=-1)

def rgb_to_jnd_kkk(rgb, converter, jnd_model):
    rgb_np = rgb.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float64)
    out = []
    for i in range(rgb_np.shape[0]):
        img = converter.normalize_image(rgb_np[i].copy())
        lin = converter.Inverse_sRGB_Companding(img)
        XYZ = converter.Linear_RGB_to_XYZ(lin)
        xyL = converter.XYZ_to_xyY(XYZ, L_MAX)
        L_phys = L_MIN + (L_MAX - L_MIN) * xyL[..., 2]
        jnd_model.find_La(L_phys)
        jnd_model.build_level_boundaries()
        k = jnd_model.L_to_k(L_phys).astype(np.float32)
        out.append(np.stack([k, k, k], axis=-1))
    arr = np.stack(out, 0).astype(np.float32)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).to(rgb.device)

# обучение jnd_kkk на STL-10 
def precompute_jnd(split, converter, jnd_model, device):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"jnd_kkk_{split}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return torch.tensor(d["x"]), torch.tensor(d["y"])
    stl = load_dataset("jxie/stl10")[split]
    ds = STL10RGBDataset(stl, transform=transforms.ToTensor())
    ld = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4)
    xs, ys = [], []
    for rgb, y in tqdm(ld, desc=f"jnd precompute [{split}]"):
        xs.append(rgb_to_jnd_kkk(rgb.to(device), converter, jnd_model).cpu())
        ys.append(y)
    X, Y = torch.cat(xs), torch.cat(ys)
    np.savez(cache, x=X.numpy(), y=Y.numpy())
    return X, Y


class AugJND(torch.utils.data.Dataset):
    def __init__(self, X, Y, train):
        self.X, self.Y, self.train = X, Y, train

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        img, y = self.X[i], self.Y[i]
        if self.train:
            if torch.rand(1) > 0.5:
                img = TF.hflip(img)
            img = TF.pad(img, 4, padding_mode="reflect")
            a, b, h, w = transforms.RandomCrop.get_params(img, (img.shape[1] - 8, img.shape[2] - 8))
            img = TF.crop(img, a, b, self.X.shape[2], self.X.shape[3])
        return img, y


def train_jnd_model(device):
    converter, jnd_model = ImageConverter(), JNDModel(L_MIN, L_MAX)
    Xtr, Ytr = precompute_jnd("train", converter, jnd_model, device)
    Xte, Yte = precompute_jnd("test", converter, jnd_model, device)

    tr = DataLoader(AugJND(Xtr, Ytr, True), batch_size=128, shuffle=True, num_workers=4)
    te = DataLoader(TensorDataset(Xte, Yte), batch_size=256, shuffle=False, num_workers=4)

    model = make_backbone().to(device)
    opt = optim.AdamW(model.parameters(), lr=JND_LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=JND_EPOCHS)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    os.makedirs("models", exist_ok=True)
    for ep in range(1, JND_EPOCHS + 1):
        model.train()
        for x, y in tqdm(tr, desc=f"jnd E{ep}/{JND_EPOCHS}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        sched.step()
        model.eval()
        c = t = 0
        with torch.no_grad():
            for x, y in te:
                x, y = x.to(device), y.to(device)
                c += (model(x).argmax(1) == y).sum().item()
                t += y.size(0)
        acc = c / t
        if acc > best:
            best = acc
            torch.save(model.state_dict(), JND_CKPT)
        print(f"  jnd_kkk ep{ep:02d} clean={acc:.4f}")
    model.load_state_dict(torch.load(JND_CKPT, map_location=device))
    print(f"jnd_kkk обучена, best clean={best:.4f}")
    return model


# BPDA-PGD для недифференцируемого JND
def bpda_pgd_acc(model, loader, eps, device, steps=PGD_STEPS):
    converter, jnd_model = ImageConverter(), JNDModel(L_MIN, L_MAX)
    model.eval()
    alpha = eps * 2.5 / steps
    c = t = 0
    for x, y in tqdm(loader, desc=f"BPDA-PGD eps={eps*255:.0f}", leave=False):
        x, y = x.to(device), y.to(device)
        x0 = x.clone()
        x_adv = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1)
        for _ in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            jnd = rgb_to_jnd_kkk(x_adv, converter, jnd_model)
            x_ste = jnd.detach() + (x_adv - x_adv.detach())
            with torch.enable_grad():
                loss = F.cross_entropy(model(x_ste), y)
            g = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + alpha * g.sign()
            x_adv = torch.min(torch.max(x_adv, x0 - eps), x0 + eps).clamp(0, 1)
        with torch.no_grad():
            jnd = rgb_to_jnd_kkk(x_adv, converter, jnd_model)
            c += (model(jnd).argmax(1) == y).sum().item()
            t += y.size(0)
    return c / t


@torch.no_grad()
def clean_acc_jnd(model, loader, device):
    converter, jnd_model = ImageConverter(), JNDModel(L_MIN, L_MAX)
    model.eval()
    c = t = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        c += (model(rgb_to_jnd_kkk(x, converter, jnd_model)).argmax(1) == y).sum().item()
        t += y.size(0)
    return c / t


# общий RGB-суррогат для transfer-переноса
def train_rgb_surrogate(device):
    if os.path.exists(RGB_CKPT):
        m = make_backbone().to(device)
        m.load_state_dict(torch.load(RGB_CKPT, map_location=device))
        return m.eval()
    tf_tr = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(96, padding=8),
                                transforms.ToTensor()])
    stl = load_dataset("jxie/stl10")
    tr = DataLoader(STL10RGBDataset(stl["train"], transform=tf_tr), batch_size=128,
                    shuffle=True, num_workers=4, drop_last=True)
    te = DataLoader(STL10RGBDataset(stl["test"], transform=transforms.ToTensor()),
                    batch_size=256, shuffle=False, num_workers=4)
    m = make_backbone().to(device)
    opt = optim.AdamW(m.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=JND_EPOCHS)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    os.makedirs("models", exist_ok=True)
    for ep in range(1, JND_EPOCHS + 1):
        m.train()
        for x, y in tqdm(tr, desc=f"rgb E{ep}/{JND_EPOCHS}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(m(x), y).backward(); opt.step()
        sched.step()
        m.eval(); c = t = 0
        with torch.no_grad():
            for x, y in te:
                x, y = x.to(device), y.to(device)
                c += (m(x).argmax(1) == y).sum().item(); t += y.size(0)
        if c / t > best:
            best = c / t; torch.save(m.state_dict(), RGB_CKPT)
    m.load_state_dict(torch.load(RGB_CKPT, map_location=device))
    print(f"RGB-суррогат обучен, best clean={best:.4f}")
    return m.eval()


def _pgd_surrogate(surrogate, eps, steps=PGD_STEPS):
    return torchattacks.PGD(surrogate, eps=eps, alpha=eps * 2.5 / steps,
                            steps=steps, random_start=True)


def transfer_acc_jnd(jnd_net, surrogate, loader, eps, device):
    conv, jm = ImageConverter(), JNDModel(L_MIN, L_MAX)
    atk = _pgd_surrogate(surrogate, eps)
    c = t = 0
    for x, y in tqdm(loader, desc=f"transfer->jnd eps={eps*255:.0f}", leave=False):
        x, y = x.to(device), y.to(device)
        x_adv = atk(x, y)
        with torch.no_grad():
            c += (jnd_net(rgb_to_jnd_kkk(x_adv, conv, jm)).argmax(1) == y).sum().item()
            t += y.size(0)
    return c / t


def transfer_acc_mine(my01, surrogate, loader, eps, device, reps=10):
    atk = _pgd_surrogate(surrogate, eps)
    c = t = 0
    for x, y in tqdm(loader, desc=f"transfer->my eps={eps*255:.0f}", leave=False):
        x, y = x.to(device), y.to(device)
        x_adv = atk(x, y)
        with torch.no_grad():
            logits = sum(my01(x_adv) for _ in range(reps)) / reps    # усреднение по шуму
            c += (logits.argmax(1) == y).sum().item(); t += y.size(0)
    return c / t


# ============ атаки, возвращающие x_adv (для унифицированных метрик) ============
def bpda_pgd_delta(jnd_net, x, y, eps, device, steps=PGD_STEPS):
    conv, jm = ImageConverter(), JNDModel(L_MIN, L_MAX)
    jnd_net.eval()
    alpha = eps * 2.5 / steps
    x0 = x.clone()
    xa = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        jnd = rgb_to_jnd_kkk(xa, conv, jm)
        x_ste = jnd.detach() + (xa - xa.detach())          # BPDA straight-through
        with torch.enable_grad():
            loss = F.cross_entropy(jnd_net(x_ste), y)
        g = torch.autograd.grad(loss, xa)[0]
        xa = (xa.detach() + alpha * g.sign())
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def eot_pgd_delta(model01, x, y, eps, device, steps=PGD_STEPS, eot=EOT_K):
    model01.requires_grad_(False)
    alpha = eps * 2.5 / steps
    x0 = x.clone()
    xa = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        g = torch.zeros_like(xa)
        for _ in range(max(1, eot)):
            xv = xa.detach().requires_grad_(True)
            with torch.enable_grad():
                loss = F.cross_entropy(model01(xv), y)
            g += torch.autograd.grad(loss, xv)[0]
        g /= max(1, eot)
        xa = (xa.detach() + alpha * g.sign())
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


# ============ унифицированные метрики ============
def _lp(x, x_adv):
    d = (x_adv - x).flatten(1)
    return (d.norm(1, 1).mean().item(),          # L1
            d.norm(2, 1).mean().item(),          # L2
            d.abs().amax(1).mean().item())       # Linf (должно быть ≤ eps — проверка корректности)


def eval_metrics(predict, attack, loader, device, reps=1, jnd_tf=None):
    """predict(x)->logits, attack(x,y)->x_adv в [0,1]. reps — усреднение предсказания по шуму.
    jnd_tf(z)->JND-представление: если задан, Lp считается и в JND-домене (перцептивная видимость)."""
    cc = ac = flip = corr = tot = 0
    L1 = L2 = Li = 0.0
    jL1 = jL2 = jLi = 0.0; nb = 0
    for x, y in tqdm(loader, desc="metrics", leave=False):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pc = (sum(predict(x) for _ in range(reps)) / reps).argmax(1)
        x_adv = attack(x, y)
        with torch.no_grad():
            pa = (sum(predict(x_adv) for _ in range(reps)) / reps).argmax(1)
        ok_c, ok_a = (pc == y), (pa == y)
        cc += ok_c.sum().item(); ac += ok_a.sum().item(); tot += y.size(0)
        corr += ok_c.sum().item(); flip += (ok_c & ~ok_a).sum().item()
        a, b, c = _lp(x, x_adv); L1 += a; L2 += b; Li += c
        if jnd_tf is not None:
            with torch.no_grad():
                ja, jb, jc = _lp(jnd_tf(x), jnd_tf(x_adv))     # расстояние в JND-домене
            jL1 += ja; jL2 += jb; jLi += jc
        nb += 1
    out = dict(clean=round(cc / tot, 3), robust=round(ac / tot, 3),
               ASR=round(flip / max(1, corr), 3),
               L1=round(L1 / nb, 2), L2=round(L2 / nb, 3), Linf=round(Li / nb, 4))
    if jnd_tf is not None:
        out.update(jnd_L1=round(jL1 / nb, 2), jnd_L2=round(jL2 / nb, 3), jnd_Linf=round(jLi / nb, 3))
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    conv, jm = ImageConverter(), JNDModel(L_MIN, L_MAX)

    jnd = make_backbone().to(device)
    if os.path.exists(JND_CKPT):
        jnd.load_state_dict(torch.load(JND_CKPT, map_location=device)); jnd.eval()
        print(f"jnd_kkk загружена из {JND_CKPT}")
    else:
        jnd = train_jnd_model(device)
    surrogate = train_rgb_surrogate(device)
    my = load_model(MY_CKPT, noise_sigma=MY_NOISE, device=device, **MY_KW)

    predict_jnd = lambda x: jnd(rgb_to_jnd_kkk(x, conv, jm))
    predict_my = lambda x: my(x)
    jnd_tf = (lambda z: rgb_to_jnd_kkk(z, conv, jm)) if JND_DIST else None

    ld = get_loader01(n=N_EVAL, batch_size=128, seed=0)

    rows = []
    for e in EPS_LIST:
        surr = _pgd_surrogate(surrogate, e)
        # (модель, подход, predict, attack->x_adv, reps предсказания)
        runs = [
            ('jnd_kkk', 'white-box', predict_jnd, lambda x, y, e=e: bpda_pgd_delta(jnd, x, y, e, device), 1),
            ('jnd_kkk', 'transfer',  predict_jnd, lambda x, y, a=surr: a(x, y),                            1),
            ('mine',    'white-box', predict_my,  lambda x, y, e=e: eot_pgd_delta(my, x, y, e, device),    10),
            ('mine',    'transfer',  predict_my,  lambda x, y, a=surr: a(x, y),                            10),
        ]
        for name, appr, predict, attack, reps in runs:
            m = eval_metrics(predict, attack, ld, device, reps=reps, jnd_tf=jnd_tf)
            m.update(model=name, approach=appr, eps=f'{e*255:.0f}/255')
            rows.append(m)
            print(f"{name:8s} {appr:9s} eps={e*255:>2.0f}/255  clean={m['clean']} robust={m['robust']} "
                  f"ASR={m['ASR']} L1={m['L1']} L2={m['L2']} Linf={m['Linf']}", flush=True)

    cols = ['model', 'approach', 'eps', 'clean', 'robust', 'ASR', 'L1', 'L2', 'Linf']
    if JND_DIST:
        cols += ['jnd_L1', 'jnd_L2', 'jnd_Linf']
    df = pd.DataFrame(rows)[cols]
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/jnd_unified_metrics.csv", index=False)
    print("\n" + df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
