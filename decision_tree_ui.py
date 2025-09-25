# Імпортуємо необхідні бібліотеки
import tkinter as tk                           # базова бібліотека для GUI
from tkinter import ttk                        # сучасні віджети Tkinter (кнопки, вкладки, рамки)
from tkinter.scrolledtext import ScrolledText  # текстове поле з прокруткою
import matplotlib.pyplot as plt                # для побудови графіків
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# FigureCanvasTkAgg — вставляє графік matplotlib у Tkinter
# NavigationToolbar2Tk — додає панель інструментів (Zoom, Pan, Reset)

from sklearn.model_selection import train_test_split   # для поділу даних на train/test
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
# DecisionTreeClassifier — модель дерева рішень
# export_text — вивід правил дерева у текстовому вигляді
# plot_tree — малювання дерева у вигляді графіка

from sklearn.metrics import accuracy_score             # для оцінки точності моделі


# Основна функція, яка будує вкладку з деревом рішень
def build_decision_tree_tab(nb, df):
    # Створюємо нову вкладку (frame) у Notebook
    frame = ttk.Frame(nb)
    nb.add(frame, text="Дерево рішень")

    # Заголовок вкладки
    ttk.Label(
        frame,
        text="Дерево рішень — прогноз навантаження",
        font=("Arial", 16, "bold"),
        foreground="navy"
    ).pack(pady=10)

    # Якщо дані порожні або немає колонки load_mw — показуємо повідомлення
    if df.empty or "load_mw" not in df.columns:
        ttk.Label(frame, text="Немає даних для дерева").pack(pady=20)
        return

    # --- Формування цільової змінної ---
    median_load = df["load_mw"].median()   # знаходимо медіану навантаження
    df = df.copy()                         # робимо копію, щоб не псувати оригінал
    # створюємо нову колонку: 1 = високе навантаження, 0 = низьке
    df["high_load"] = (df["load_mw"] >= median_load).astype(int)

    # Вибираємо ознаки (features), які будемо використовувати
    features = [c for c in ["temp_c", "wind_mps", "is_holiday"] if c in df.columns]
    if not features:   # якщо немає потрібних ознак — повідомляємо
        ttk.Label(frame, text="Відсутні необхідні ознаки").pack(pady=20)
        return

    X = df[features]          # матриця ознак
    y = df["high_load"]       # цільова змінна (0 або 1)

    # --- Навчання дерева ---
    # Ділимо дані на навчальні та тестові (70% / 30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Створюємо модель дерева рішень
    clf = DecisionTreeClassifier(
        criterion="gini",   # критерій розщеплення (індекс Джині)
        max_depth=4,        # максимальна глибина дерева
        random_state=42     # фіксуємо випадковість для відтворюваності
    )
    clf.fit(X_train, y_train)   # навчаємо модель
    acc = accuracy_score(y_test, clf.predict(X_test))  # обчислюємо точність

    # --- Ліва частина: текст з прокруткою ---
    left = ttk.Frame(frame)
    left.pack(side="left", fill="both", expand=True, padx=10, pady=10)

    # Створюємо текстове поле з прокруткою
    text = ScrolledText(left, wrap="word", font=("Consolas", 12))
    text.pack(fill="both", expand=True)

    # Налаштовуємо стилі для тексту
    text.tag_configure("bold", font=("Consolas", 12, "bold"))
    text.tag_configure("header", font=("Arial", 13, "bold"), foreground="darkgreen")
    text.tag_configure("highlight", font=("Consolas", 12, "bold"), foreground="red")

    # Виводимо результати
    text.insert("end", "=== Результати дерева рішень ===\n\n", "header")
    text.insert("end", "Точність моделі: ", "bold")
    text.insert("end", f"{acc:.3f}\n\n", "highlight")

    # Отримуємо правила дерева у текстовому вигляді
    rules_text = export_text(clf, feature_names=features)
    # Замінюємо "class: 0/1" на зрозумілі назви
    rules_text = rules_text.replace("class: 0", "→ Клас: Низьке навантаження")
    rules_text = rules_text.replace("class: 1", "→ Клас: Високе навантаження")

    text.insert("end", "=== Правила дерева ===\n", "header")
    text.insert("end", rules_text + "\n\n")

    # Додаємо пояснення для захисту
    text.insert("end", "=== Пояснення для захисту ===\n", "header")
    text.insert("end",
        "1. Точність моделі показує, наскільки добре дерево прогнозує.\n"
        "2. Правила дерева — це умови, за якими система приймає рішення.\n"
        "3. У вузлах показано gini (чистота), samples (кількість прикладів),\n"
        "   value (розподіл класів) і class (результат).\n"
        "4. Кольори вузлів допомагають швидко зрозуміти, який клас переважає.\n"
        "5. Дерево можна масштабувати та прокручувати для детального аналізу.\n"
    )

    # --- Права частина: графік з прокруткою ---
    right = ttk.Frame(frame)
    right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    # Створюємо Canvas з горизонтальним і вертикальним скролом
    canvas_frame = tk.Canvas(right)
    scroll_y = tk.Scrollbar(right, orient="vertical", command=canvas_frame.yview)
    scroll_x = tk.Scrollbar(right, orient="horizontal", command=canvas_frame.xview)

    scroll_y.pack(side="right", fill="y")
    scroll_x.pack(side="bottom", fill="x")
    canvas_frame.pack(side="left", fill="both", expand=True)

    canvas_frame.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

    inner = ttk.Frame(canvas_frame)
    canvas_frame.create_window((0,0), window=inner, anchor="nw")

    # Малюємо дуже широке дерево (щоб було читабельно)
    fig, ax = plt.subplots(figsize=(40,8))  # ширина=40, висота=8
    plot_tree(
        clf,
        feature_names=features,
        class_names=["Низьке навантаження", "Високе навантаження"],
        filled=True, rounded=True,
        proportion=False,
        fontsize=9,
        ax=ax
    )
    ax.set_title("Дерево рішень (широкий вигляд)", fontsize=12, fontweight="bold")
    fig.tight_layout()

    # Вставляємо графік у Tkinter
    canvas = FigureCanvasTkAgg(fig, master=inner)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # Додаємо панель інструментів (Zoom, Pan, Reset)
    toolbar_frame = ttk.Frame(inner)
    toolbar_frame.pack(fill="x")
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()

    # Оновлюємо область прокрутки
    inner.update_idletasks()
    canvas_frame.config(scrollregion=canvas_frame.bbox("all"))

    # Закриваємо фігуру matplotlib (щоб не дублювалась у пам’яті)
    plt.close(fig)