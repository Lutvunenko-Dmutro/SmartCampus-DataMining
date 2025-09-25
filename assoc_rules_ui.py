import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Функція для побудови вкладки "Асоціативні правила"
def build_assoc_rules_tab(nb, df):
    # Створюємо новий фрейм (контейнер) у Notebook
    frame = ttk.Frame(nb)
    # Додаємо цей фрейм як нову вкладку з назвою
    nb.add(frame, text="Асоціативні правила")

    # Заголовок вкладки
    ttk.Label(frame,
              text="Асоціативні правила — результати аналізу",
              font=("Arial", 16, "bold"),   # великий жирний шрифт
              foreground="navy"             # темно-синій колір
              ).pack(pady=10)               # відступ зверху/знизу

    # Якщо DataFrame порожній — повідомляємо
    if df.empty:
        ttk.Label(frame, text="Немає даних для аналізу").pack(pady=20)
        return

    # --- Дискретизація (перетворення числових даних у категорії) ---
    disc = pd.DataFrame()
    if "temp_c" in df.columns:
        # Температура: розбиваємо на 3 категорії
        disc["Температура"] = pd.cut(df["temp_c"],
                                     bins=[-100,0,10,100],
                                     labels=["холодно","помірно","тепло"])
    if "wind_mps" in df.columns:
        # Вітер: слабкий/середній/сильний
        disc["Вітер"] = pd.cut(df["wind_mps"],
                               bins=[-1,3,7,100],
                               labels=["слабкий","середній","сильний"])
    if "is_holiday" in df.columns:
        # Свято: 0 → робочий день, 1 → свято
        disc["Свято"] = df["is_holiday"].map({0:"робочий день",1:"свято"})
    if "load_mw" in df.columns:
        # Навантаження: вище/нижче медіани
        median_load = df["load_mw"].median()
        disc["Навантаження"] = df["load_mw"].apply(
            lambda v: "високе" if v>=median_load else "низьке"
        )

    # Якщо після дискретизації нічого не вийшло
    if disc.empty:
        ttk.Label(frame, text="Недостатньо колонок для побудови правил").pack(pady=20)
        return

    # --- Побудова асоціативних правил ---
    onehot = pd.get_dummies(disc)  # перетворюємо категорії у one-hot формат
    freq = apriori(onehot, min_support=0.1, use_colnames=True)  # шукаємо часті набори
    rules = association_rules(freq, metric="confidence", min_threshold=0.6)  # генеруємо правила
    if rules.empty:
        ttk.Label(frame, text="Правила не знайдено").pack(pady=20)
        return
    # Сортуємо правила за lift (сила зв’язку)
    rules = rules.sort_values(by="lift", ascending=False)

    # --- Ліва частина: текст з прокруткою ---
    left = ttk.Frame(frame)
    left.pack(side="left", fill="both", expand=True, padx=10, pady=10)

    text = ScrolledText(left, wrap="word", font=("Consolas", 12))
    text.pack(fill="both", expand=True)

    text.insert("end", "📊 Топ-20 асоціативних правил\n\n")

    # Функція для форматування множин у читабельний вигляд
    def fmt(itemset):
        return ", ".join([s.replace("_", " = ") for s in itemset])

    # Виводимо перші 20 правил у текстовому вигляді
    for i, (_, row) in enumerate(rules.head(20).iterrows(), start=1):
        text.insert("end", f"{i}. {fmt(row['antecedents'])} → {fmt(row['consequents'])}\n")
        text.insert("end", f"   support={row['support']:.2f}, "
                           f"conf={row['confidence']:.2f}, "
                           f"lift={row['lift']:.2f}\n\n")

    # Додаємо пояснення метрик
    text.insert("end", "support = частота, confidence = надійність, lift = сила зв’язку\n")

    # --- Права частина: таблиця з ползунками ---
    right = ttk.Frame(frame)
    right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    # Контейнер для Canvas + Scrollbars
    table_container = ttk.Frame(right)
    table_container.pack(fill="both", expand=True)

    canvas = tk.Canvas(table_container)
    canvas.pack(side="left", fill="both", expand=True)

    # Вертикальний скролбар
    vscroll = tk.Scrollbar(table_container, orient="vertical", command=canvas.yview)
    vscroll.pack(side="right", fill="y")

    # Горизонтальний скролбар
    hscroll = tk.Scrollbar(right, orient="horizontal", command=canvas.xview)
    hscroll.pack(side="bottom", fill="x")

    # Прив’язуємо скролбари до Canvas
    canvas.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)

    # Вставляємо таблицю у Canvas
    table_frame = ttk.Frame(canvas)
    canvas.create_window((0,0), window=table_frame, anchor="nw")

    # Заголовки таблиці
    headers = ["ANTEСEDENTS", "CONSEQUENTS", "SUPPORT", "CONFIDENCE", "LIFT"]
    for j, h in enumerate(headers):
        tk.Label(table_frame, text=h,
                 font=("Arial", 12, "bold"),
                 fg="white", bg="navy",          # білий текст на синьому фоні
                 borderwidth=1, relief="solid",  # рамка
                 padx=8, pady=5                  # відступи
                 ).grid(row=0, column=j, sticky="nsew")

    # Заповнення таблиці з чергуванням кольорів рядків
    for i, (_, row) in enumerate(rules.head(20).iterrows(), start=1):
        antecedents = fmt(row["antecedents"])
        consequents = fmt(row["consequents"])
        values = [antecedents, consequents,
                  f"{row['support']:.2f}",
                  f"{row['confidence']:.2f}",
                  f"{row['lift']:.2f}"]

        # Чередування кольору рядків: сірий/білий
        bg_color = "#f9f9f9" if i % 2 == 0 else "white"

        for j, val in enumerate(values):
            tk.Label(table_frame, text=val,
                     font=("Consolas", 11),
                     borderwidth=1, relief="solid",
                     anchor="w", justify="left",  # вирівнювання вліво
                     wraplength=600,              # перенос рядків
                     padx=8, pady=5,
                     bg=bg_color                  # фон рядка
                     ).grid(row=i, column=j, sticky="nsew")

    # Автоматичне розтягування колонок
    for j in range(len(headers)):
        table_frame.grid_columnconfigure(j, weight=1)

    # Оновлюємо область прокрутки Canvas
    table_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))