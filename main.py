"""
Animals10 Image Classifier — Tkinter GUI
Replica exactamente el pipeline del notebook de entrenamiento:
  · HOG: cv2.HOGDescriptor, winSize=128x128, blockStride=8x8  -> 8100 features
  · LBP: skimage uniform, radius=3, equalizeHist              -> 26 features
  · MLP: NeuralNet(fc1->ReLU->drop->fc2->ReLU->drop->fc3)
  · Clases ordenadas alfabeticamente (comportamiento de LabelEncoder)

IMPORTANTE: Los modelos fueron entrenados con StandardScaler aplicado.
Si las predicciones son inconsistentes, guarda tambien el scaler durante el
entrenamiento (joblib.dump(scaler, 'scaler_hog.pkl')) y cargalo aqui.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import torch
import torch.nn as nn
import os

# ---------------------------------------------------------------------------
# Clases del dataset Animals10 -- ordenadas alfabeticamente (LabelEncoder)
# ---------------------------------------------------------------------------
CLASSES = sorted([
    "butterfly", "cat", "chicken", "cow", "dog",
    "elephant", "horse", "sheep", "spyder", "squirrel"
])
# Resultado: ['butterfly','cat','chicken','cow','dog','elephant','horse','sheep','spyder','squirrel']

CLASS_EMOJIS = {
    "butterfly": "🦋", "cat": "🐈", "chicken": "🐓", "cow": "🐄",
    "dog": "🐕", "elephant": "🐘", "horse": "🐎", "sheep": "🐑",
    "spyder": "🕷️", "squirrel": "🐿️",
}

# ---------------------------------------------------------------------------
# Arquitectura MLP -- identica a NeuralNet del notebook
# ---------------------------------------------------------------------------
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super().__init__()
        self.fc1      = nn.Linear(input_size, hidden_size1)
        self.relu1    = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2      = nn.Linear(hidden_size1, hidden_size2)
        self.relu2    = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3      = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.fc3(x)


# ---------------------------------------------------------------------------
# HOG: cv2.HOGDescriptor exacto del notebook
# winSize=128x128, blockSize=16x16, blockStride=8x8, cellSize=8x8,
# nbins=9, gammaCorrection=True, L2-Hys
# Salida: (8100,)
# ---------------------------------------------------------------------------
_HOG_DESCRIPTOR = cv2.HOGDescriptor(
    _winSize=(128, 128),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
    _gammaCorrection=True,
    _histogramNormType=cv2.HOGDescriptor_L2Hys,
)

def extract_hog(bgr_image: np.ndarray) -> np.ndarray:
    """
    Extrae HOG de una imagen BGR uint8 128x128.
    Convierte a escala de grises antes del descriptor.
    Retorna float32 de dimension 8100.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    features = _HOG_DESCRIPTOR.compute(gray).flatten()
    return features.astype(np.float32)


# ---------------------------------------------------------------------------
# LBP: skimage + equalizeHist (igual que el notebook)
# radius=3, n_points=24, method='uniform'
# Salida: (26,)
# ---------------------------------------------------------------------------
def extract_lbp(bgr_image: np.ndarray) -> np.ndarray:
    """
    Extrae LBP de una imagen BGR uint8 128x128.
    Aplica equalizeHist igual que el notebook de entrenamiento.
    Retorna histograma normalizado float32 de dimension 26.
    """
    from skimage.feature import local_binary_pattern

    gray    = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)           # clave: igual que el notebook

    LBP_RADIUS  = 3
    LBP_POINTS  = 8 * LBP_RADIUS              # 24
    lbp = local_binary_pattern(gray_eq, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_POINTS + 2                   # 26 bins (uniform LBP)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                           range=(0, n_bins), density=True)
    return hist.astype(np.float32)


# ---------------------------------------------------------------------------
# Preprocesamiento: PIL RGB -> BGR uint8 128x128
# ---------------------------------------------------------------------------
def preprocess_to_bgr128(pil_image: Image.Image) -> np.ndarray:
    """Replica cv2.imread + cv2.resize del notebook."""
    rgb = np.array(pil_image.convert("RGB").resize((128, 128)))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Carga de modelo desde checkpoint
# ---------------------------------------------------------------------------
def load_model(path: str) -> NeuralNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = NeuralNet(
        input_size=ckpt["input_size"],
        hidden_size1=ckpt["hidden_size1"],
        hidden_size2=ckpt["hidden_size2"],
        num_classes=ckpt["output_size"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Prediccion
# ---------------------------------------------------------------------------
def predict(model: NeuralNet,
            features: np.ndarray) -> tuple[str, float, np.ndarray]:
    """Retorna (clase_predicha, confianza, array_de_probabilidades)."""
    x = torch.tensor(features).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
    idx = int(probs.argmax())
    return CLASSES[idx], float(probs[idx]), probs


# ---------------------------------------------------------------------------
# Paleta visual: dark terminal / cientifico
# ---------------------------------------------------------------------------
BG       = "#0d1117"
SURFACE  = "#161b22"
CARD     = "#1f2937"
BORDER   = "#30363d"
ACCENT   = "#58a6ff"
ACCENT2  = "#3fb950"
WARN     = "#d29922"
TEXT     = "#e6edf3"
TEXT_DIM = "#7d8590"
SUCCESS  = "#3fb950"

FM = "Courier New"
FONT_TITLE  = (FM, 18, "bold")
FONT_LABEL  = (FM, 9, "bold")
FONT_BODY   = (FM, 9)
FONT_SMALL  = (FM, 8)
FONT_RESULT = (FM, 26, "bold")
FONT_CONF   = (FM, 10)


# ---------------------------------------------------------------------------
# Aplicacion principal
# ---------------------------------------------------------------------------
class App(tk.Tk):
    MODEL_PATHS = {
        "MLP-HOG": "mejor_modelo_hog.pth",
        "MLP-LBP": "mejor_modelo_lbp.pth",
    }

    def __init__(self):
        super().__init__()
        self.title("Animals10 - Clasificador de Imagenes")
        self.geometry("880x660")
        self.configure(bg=BG)
        self.resizable(False, False)

        self._pil_image: Image.Image | None = None
        self._bgr128:    np.ndarray | None  = None
        self._model_cache: dict[str, NeuralNet] = {}
        self._syncing = False

        self._apply_style()
        self._build_ui()

    # -- ttk dark theme ------------------------------------------------------
    def _apply_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TCombobox",
                    fieldbackground=SURFACE, background=SURFACE,
                    foreground=TEXT, selectbackground=ACCENT,
                    selectforeground=BG, bordercolor=BORDER,
                    arrowcolor=ACCENT, relief="flat", padding=5)
        s.map("TCombobox",
              fieldbackground=[("readonly", SURFACE)],
              foreground=[("readonly", TEXT)],
              background=[("readonly", SURFACE)])

    # -- Layout principal ----------------------------------------------------
    def _build_ui(self):
        # Encabezado
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(16, 0))
        tk.Label(hdr, text="◈ ANIMALS10", font=FONT_TITLE,
                 fg=ACCENT, bg=BG).pack(side="left")
        tk.Label(hdr, text="  clasificador HOG · LBP · MLP",
                 font=FONT_SMALL, fg=TEXT_DIM, bg=BG).pack(
            side="left", pady=(7, 0))

        # Separador
        sep = tk.Canvas(self, height=1, bg=BG, highlightthickness=0)
        sep.pack(fill="x", padx=20, pady=(6, 8))
        sep.create_line(0, 0, 900, 0, fill=BORDER)

        # Cuerpo principal
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20)

        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))

        right = tk.Frame(body, bg=BG, width=300)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self._build_image_panel(left)
        self._build_config_panel(right)
        self._build_result_panel(right)

    # -- Tarjeta generica ----------------------------------------------------
    def _card(self, parent: tk.Widget, label: str) -> tk.Frame:
        outer = tk.Frame(parent, bg=BORDER)
        outer.pack(fill="x", pady=5)
        inner = tk.Frame(outer, bg=CARD)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        tk.Label(inner, text=label, font=FONT_LABEL,
                 fg=ACCENT, bg=CARD, anchor="w").pack(
            fill="x", padx=12, pady=(10, 3))
        div = tk.Canvas(inner, height=1, bg=CARD, highlightthickness=0)
        div.pack(fill="x", padx=12, pady=(0, 8))
        div.bind("<Configure>",
                 lambda e: div.create_line(0, 0, e.width, 0, fill=BORDER))
        return inner

    # -- Panel: imagen -------------------------------------------------------
    def _build_image_panel(self, parent: tk.Frame):
        card = self._card(parent, "01 · IMAGEN")

        self._canvas = tk.Canvas(card, width=420, height=320,
                                 bg=SURFACE, highlightthickness=1,
                                 highlightbackground=BORDER)
        self._canvas.pack(padx=12, pady=(0, 8))
        self._draw_placeholder()

        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(row, text="preproc:", font=FONT_SMALL,
                 fg=TEXT_DIM, bg=CARD).pack(side="left")
        self._preproc_var = tk.StringVar(value="—")
        tk.Label(row, textvariable=self._preproc_var, font=FONT_SMALL,
                 fg=ACCENT2, bg=CARD).pack(side="left", padx=4)

        tk.Button(card, text="  CARGAR IMAGEN",
                  font=FONT_LABEL, fg=BG, bg=ACCENT,
                  activeforeground=BG, activebackground=ACCENT2,
                  bd=0, cursor="hand2", padx=18, pady=8,
                  command=self._load_image).pack(pady=(4, 12))

    # -- Panel: configuracion ------------------------------------------------
    def _build_config_panel(self, parent: tk.Frame):
        card = self._card(parent, "02 · CONFIGURACION")

        tk.Label(card, text="Descriptor de caracteristicas:",
                 font=FONT_SMALL, fg=TEXT_DIM, bg=CARD, anchor="w").pack(
            fill="x", padx=12, pady=(0, 3))
        self._desc_var = tk.StringVar(value="HOG")
        ttk.Combobox(card, textvariable=self._desc_var,
                     values=["HOG", "LBP"],
                     state="readonly", font=FONT_BODY).pack(
            padx=12, pady=(0, 10), fill="x")

        tk.Label(card, text="Modelo de clasificacion:",
                 font=FONT_SMALL, fg=TEXT_DIM, bg=CARD, anchor="w").pack(
            fill="x", padx=12, pady=(0, 3))
        self._model_var = tk.StringVar(value="MLP-HOG")
        ttk.Combobox(card, textvariable=self._model_var,
                     values=["MLP-HOG", "MLP-LBP"],
                     state="readonly", font=FONT_BODY).pack(
            padx=12, pady=(0, 8), fill="x")

        # Sincronizacion automatica descriptor <-> modelo
        self._desc_var.trace_add("write", self._sync_desc_to_model)
        self._model_var.trace_add("write", self._sync_model_to_desc)

        # Nota sobre StandardScaler
        note = tk.Frame(card, bg="#1a2235", highlightbackground=WARN,
                        highlightthickness=1)
        note.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(note,
                 text=(
                     "Nota: el entrenamiento aplico\n"
                     "StandardScaler a las features.\n"
                     "Guarda scaler_hog.pkl / scaler_lbp.pkl\n"
                     "con joblib para maxima precision."
                 ),
                 font=FONT_SMALL, fg=WARN, bg="#1a2235",
                 justify="left", wraplength=250).pack(padx=8, pady=6)

        tk.Button(card, text="▶  CLASIFICAR",
                  font=FONT_LABEL, fg=BG, bg=ACCENT2,
                  activeforeground=BG, activebackground=ACCENT,
                  bd=0, cursor="hand2", pady=10,
                  command=self._classify).pack(
            fill="x", padx=12, pady=(0, 12))

    # -- Panel: resultado ----------------------------------------------------
    def _build_result_panel(self, parent: tk.Frame):
        card = self._card(parent, "03 · RESULTADO")

        self._emoji_var = tk.StringVar(value="")
        self._class_var = tk.StringVar(value="—")
        self._conf_var  = tk.StringVar(value="")
        self._info_var  = tk.StringVar(value="")

        tk.Label(card, textvariable=self._emoji_var,
                 font=("Segoe UI Emoji", 40), bg=CARD, fg=TEXT).pack(pady=(6, 0))
        tk.Label(card, textvariable=self._class_var,
                 font=FONT_RESULT, fg=SUCCESS, bg=CARD).pack()
        tk.Label(card, textvariable=self._conf_var,
                 font=FONT_CONF, fg=TEXT_DIM, bg=CARD).pack(pady=(2, 4))

        # Barra de confianza
        bar_frame = tk.Frame(card, bg=SURFACE, height=10)
        bar_frame.pack(fill="x", padx=12, pady=(0, 6))
        bar_frame.pack_propagate(False)
        self._bar = tk.Frame(bar_frame, bg=ACCENT2, height=10)
        self._bar.place(x=0, y=0, relheight=1, relwidth=0.0)

        # Top-3 predicciones
        top3_frame = tk.Frame(card, bg=CARD)
        top3_frame.pack(fill="x", padx=12, pady=(2, 4))
        self._top3: list[tk.Label] = []
        for _ in range(3):
            lbl = tk.Label(top3_frame, text="", font=FONT_SMALL,
                           fg=TEXT_DIM, bg=CARD, anchor="w")
            lbl.pack(fill="x")
            self._top3.append(lbl)

        tk.Label(card, textvariable=self._info_var, font=FONT_SMALL,
                 fg=TEXT_DIM, bg=CARD, wraplength=270,
                 justify="center").pack(pady=(2, 12))

    # -- Utilidades UI -------------------------------------------------------
    def _draw_placeholder(self):
        W, H = 420, 320
        self._canvas.delete("all")
        self._canvas.create_rectangle(0, 0, W, H, fill=SURFACE, outline="")
        for x in range(0, W, 35):
            self._canvas.create_line(x, 0, x, H, fill=BORDER)
        for y in range(0, H, 35):
            self._canvas.create_line(0, y, W, y, fill=BORDER)
        self._canvas.create_text(W//2, H//2 - 12,
                                 text="[ sin imagen ]",
                                 fill=TEXT_DIM, font=FONT_LABEL)
        self._canvas.create_text(W//2, H//2 + 12,
                                 text="carga una imagen para comenzar",
                                 fill=BORDER, font=FONT_SMALL)

    def _show_image(self, pil_img: Image.Image):
        W, H = 420, 320
        img = pil_img.copy()
        img.thumbnail((W - 16, H - 16), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        self._canvas.create_rectangle(0, 0, W, H, fill=SURFACE, outline="")
        self._canvas.create_image(W//2, H//2, image=self._tk_img, anchor="center")

    def _update_conf_bar(self, ratio: float):
        self._bar.place(relwidth=max(0.0, min(1.0, ratio)))

    def _reset_result(self):
        self._emoji_var.set("")
        self._class_var.set("—")
        self._conf_var.set("")
        self._info_var.set("")
        self._update_conf_bar(0)
        for lbl in self._top3:
            lbl.config(text="")

    def _sync_desc_to_model(self, *_):
        if self._syncing:
            return
        self._syncing = True
        self._model_var.set(
            "MLP-HOG" if self._desc_var.get() == "HOG" else "MLP-LBP")
        self._syncing = False

    def _sync_model_to_desc(self, *_):
        if self._syncing:
            return
        self._syncing = True
        self._desc_var.set(
            "HOG" if self._model_var.get() == "MLP-HOG" else "LBP")
        self._syncing = False

    # -- Acciones principales ------------------------------------------------
    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imagenes", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self._pil_image = Image.open(path).convert("RGB")
            self._bgr128    = preprocess_to_bgr128(self._pil_image)
            self._show_image(self._pil_image)
            w, h = self._pil_image.size
            self._preproc_var.set(
                f"original {w}x{h}  ->  resize 128x128  ->  BGR->gray"
            )
            self._reset_result()
        except Exception as e:
            messagebox.showerror("Error al cargar imagen", str(e))

    def _classify(self):
        if self._pil_image is None:
            messagebox.showwarning("Sin imagen",
                                   "Primero carga una imagen.")
            return

        descriptor = self._desc_var.get()
        model_key  = self._model_var.get()
        model_path = self.MODEL_PATHS[model_key]

        if not os.path.exists(model_path):
            messagebox.showerror(
                "Modelo no encontrado",
                f"No se encontro el archivo:\n  {model_path}\n\n"
                "Asegurate de ejecutar la app desde el directorio del proyecto."
            )
            return

        # Feedback inmediato
        self._class_var.set("procesando...")
        self._emoji_var.set("⏳")
        self._conf_var.set("")
        self.update_idletasks()

        try:
            # 1. Extraccion de caracteristicas
            if descriptor == "HOG":
                features  = extract_hog(self._bgr128)
                feat_info = f"HOG  |  {features.shape[0]} dim  |  cv2.HOGDescriptor"
            else:
                features  = extract_lbp(self._bgr128)
                feat_info = f"LBP  |  {features.shape[0]} dim  |  uniform r=3 + equalizeHist"

            # 2. Cargar modelo (con cache en memoria)
            if model_key not in self._model_cache:
                self._model_cache[model_key] = load_model(model_path)
            model = self._model_cache[model_key]

            # 3. Prediccion
            label, conf, probs = predict(model, features)

            # 4. Actualizar UI
            self._emoji_var.set(CLASS_EMOJIS.get(label, "?"))
            self._class_var.set(label.upper())
            self._conf_var.set(f"confianza: {conf * 100:.1f}%")
            self._update_conf_bar(conf)
            self._info_var.set(
                f"descriptor: {descriptor}  ·  modelo: {model_key}\n{feat_info}"
            )

            # Top-3 clases con mini barra de texto
            top3 = sorted(enumerate(probs), key=lambda x: -x[1])[:3]
            for i, (idx, p) in enumerate(top3):
                bar_len = int(p * 18)
                bar_str = "█" * bar_len + "░" * (18 - bar_len)
                self._top3[i].config(
                    text=f"{CLASSES[idx]:<12} {p*100:5.1f}%  {bar_str}"
                )

        except Exception as e:
            self._class_var.set("ERROR")
            self._emoji_var.set("⚠️")
            self._conf_var.set("")
            messagebox.showerror("Error en clasificacion", str(e))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = App()
    app.mainloop()