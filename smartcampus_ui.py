import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

from decision_tree_ui import build_decision_tree_tab
from assoc_rules_ui import build_assoc_rules_tab
from nonlinear_ui import build_nonlinear_tab

class SmartCampusApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SmartCampus — Аналітична панель")
        self.geometry("1280x850")

        # Стиль оформлення
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook.Tab", font=("Arial", 12, "bold"), padding=[15, 8])
        style.configure("TLabel", font=("Arial", 12))
        style.configure("Header.TLabel", font=("Arial", 16, "bold"))

        self.df = self._load_data()
        self._build_ui()

    def _load_data(self):
        try:
            return pd.read_csv("power_load_hourly.csv")
        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалося відкрити power_load_hourly.csv\n{e}")
            return pd.DataFrame()

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        # Вкладки
        build_decision_tree_tab(nb, self.df)
        build_assoc_rules_tab(nb, self.df)
        build_nonlinear_tab(nb, self.df)

if __name__ == "__main__":
    app = SmartCampusApp()
    app.mainloop()