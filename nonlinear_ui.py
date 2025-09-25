# Імпортуємо бібліотеки
import tkinter as tk                          # базова бібліотека для створення GUI
from tkinter import ttk                       # сучасні віджети Tkinter (кнопки, вкладки, рамки)
from tkinter.scrolledtext import ScrolledText # текстове поле з прокруткою
import numpy as np                             # бібліотека для роботи з масивами та математики
import matplotlib.pyplot as plt                # для побудови графіків
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# FigureCanvasTkAgg — вставляє графік matplotlib у Tkinter
from scipy import stats                        # статистичні методи (лінійна регресія)

# Функція для створення вкладки "Нелінійна регресія"
def build_nonlinear_tab(nb, df):
    frame = ttk.Frame(nb)                      # створюємо новий контейнер (frame) у Notebook
    nb.add(frame, text="Нелінійна регресія")   # додаємо вкладку з назвою

    # Заголовок вкладки
    ttk.Label(frame,
              text="Нелінійна регресія — апроксимація залежності", # текст заголовка
              font=("Arial", 16, "bold"),                         # шрифт Arial, 16 пт, жирний
              foreground="navy"                                   # темно-синій колір
              ).pack(pady=10)                                     # розміщуємо з відступом 10 пікселів

    # Перевірка даних
    if "temp_c" not in df.columns or "load_mw" not in df.columns: # якщо немає потрібних колонок
        ttk.Label(frame, text="У CSV немає колон temp_c та load_mw").pack(pady=20)
        return                                                    # вихід з функції

    data = df[["temp_c", "load_mw"]].dropna()                     # беремо тільки потрібні колонки, видаляємо NaN
    data = data[(data["temp_c"] > 0) & (data["load_mw"] > 0)]     # залишаємо тільки додатні значення
    if data.empty:                                                # якщо після фільтрації даних нема
        ttk.Label(frame, text="Недостатньо даних для регресії").pack(pady=20)
        return

    x, y = data["temp_c"].values, data["load_mw"].values          # X = температура, Y = навантаження

    # --- Степенева модель ---
    lx, ly = np.log(x), np.log(y)                                 # логарифмуємо X і Y
    slope_p, intercept_p, r_p, _, _ = stats.linregress(lx, ly)    # лінійна регресія у лог-просторі
    a_p, b_p, r2_p = np.exp(intercept_p), slope_p, r_p**2         # відновлюємо параметри степеневої моделі

    # --- Експоненціальна модель ---
    slope_e, intercept_e, r_e, _, _ = stats.linregress(x, ly)     # регресія: ln(y) від x
    a_e, b_e, r2_e = np.exp(intercept_e), slope_e, r_e**2         # параметри експоненціальної моделі

    # --- Графік ---
    fig, ax = plt.subplots(figsize=(9,6))                         # створюємо фігуру 9x6 дюймів
    ax.scatter(x, y, s=25, alpha=0.7, label="Спостережені дані", color="black") # точки даних
    xs = np.linspace(x.min(), x.max(), 200)                       # рівномірна сітка X для побудови кривих
    ax.plot(xs, a_p*xs**b_p,                                      # степенева крива
            label=f"Степенева (R²={r2_p:.3f})",                   # підпис з R²
            color="tab:blue", linewidth=2)
    ax.plot(xs, a_e*np.exp(b_e*xs), "--",                         # експоненціальна крива
            label=f"Експоненціальна (R²={r2_e:.3f})",             # підпис з R²
            color="tab:orange", linewidth=2)

    ax.set_title("Нелінійна регресія: Навантаження від температури", fontsize=13, fontweight="bold")
    ax.set_xlabel("Температура (°C)")                             # підпис осі X
    ax.set_ylabel("Навантаження (MW)")                            # підпис осі Y
    ax.grid(True, linestyle="--", alpha=0.5)                      # сітка пунктиром
    ax.legend(frameon=True, facecolor="white")                    # легенда з білим фоном

    canvas = FigureCanvasTkAgg(fig, master=frame)                 # вставляємо графік у Tkinter
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    plt.close(fig)                                                # закриваємо фігуру, щоб не дублювалась

    # --- Текстовий звіт з прокруткою ---
    report = ScrolledText(frame, wrap="word", font=("Consolas", 12), height=12)
    report.pack(fill="both", expand=True, padx=10, pady=10)

    # Стилі для тексту
    report.tag_configure("bold", font=("Consolas", 12, "bold"))   # жирний стиль
    report.tag_configure("header", font=("Arial", 13, "bold"), foreground="darkgreen") # зелений заголовок

    # Вивід результатів
    report.insert("end", "=== Степенева модель ===\n", "header")  # заголовок
    report.insert("end", f"Формула: y = {a_p:.3f} * x^{b_p:.3f}\n") # формула степеневої моделі
    report.insert("end", f"Коефіцієнт детермінації R² = {r2_p:.3f}\n\n", "bold") # R² жирним

    report.insert("end", "=== Експоненціальна модель ===\n", "header") # заголовок
    report.insert("end", f"Формула: y = {a_e:.3f} * e^({b_e:.3f}x)\n") # формула експоненціальної моделі
    report.insert("end", f"Коефіцієнт детермінації R² = {r2_e:.3f}\n", "bold") # R² жирним