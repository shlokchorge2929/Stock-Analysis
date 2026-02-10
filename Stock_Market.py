import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression

class StockPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Prediction Analysis - Pro Edition")
        self.root.geometry("1400x900")
        
        self.colors = {
            'bg_gradient_top': '#667eea',
            'bg_gradient_bottom': '#764ba2',
            'bg_main': '#1a1a2e',
            'bg_secondary': '#16213e',
            'accent_primary': '#f64c72',
            'accent_secondary': '#4facfe',
            'success': '#00f2fe',
            'warning': '#ffd93d',
            'text_primary': '#ffffff',
            'text_secondary': '#a8b2d1',
            'card_bg': '#0f3460',
            'button_gradient_1': '#f093fb',
            'button_gradient_2': '#f5576c',
        }
        
        self.root.configure(bg=self.colors['bg_main'])
        self.stock_data = None
        self.predictions = None
        
        self.create_custom_styles()
        self.create_gradient_background()
        self.create_header()
        self.create_input_section()
        self.create_info_cards()
        self.create_chart_section()
    
    def create_custom_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Custom.TCombobox',
                       fieldbackground=self.colors['card_bg'],
                       background=self.colors['accent_secondary'],
                       foreground=self.colors['text_primary'],
                       arrowcolor=self.colors['text_primary'])
    
    def create_gradient_background(self):
        self.main_canvas = tk.Canvas(self.root, bg=self.colors['bg_main'], 
                                     highlightthickness=0)
        self.main_canvas.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(self.root, orient=tk.VERTICAL, 
                                command=self.main_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.main_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.content_frame = tk.Frame(self.main_canvas, bg=self.colors['bg_main'])
        self.canvas_frame = self.main_canvas.create_window((0, 0), 
                                                           window=self.content_frame, 
                                                           anchor='nw')
        
        self.content_frame.bind('<Configure>', 
                               lambda e: self.main_canvas.configure(
                                   scrollregion=self.main_canvas.bbox('all')))
        
        self.main_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
    
    def _on_mousewheel(self, event):
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def create_header(self):
        header_frame = tk.Frame(self.content_frame, bg=self.colors['bg_main'], 
                               height=150)
        header_frame.pack(fill=tk.X, pady=(20, 10))
        
        title = tk.Label(header_frame,
                        text="STOCK PREDICTION ANALYZER",
                        font=('Helvetica', 32, 'bold'),
                        bg=self.colors['bg_main'],
                        fg=self.colors['success'])
        title.pack(pady=(20, 5))
        
        subtitle = tk.Label(header_frame,
                           text="AI-Powered Stock Market Analysis & Forecasting",
                           font=('Helvetica', 14),
                           bg=self.colors['bg_main'],
                           fg=self.colors['text_secondary'])
        subtitle.pack()
        
        line_canvas = tk.Canvas(header_frame, height=3, bg=self.colors['bg_main'], 
                               highlightthickness=0)
        line_canvas.pack(fill=tk.X, padx=100, pady=10)
        line_canvas.create_line(0, 1, 1400, 1, fill=self.colors['accent_primary'], 
                               width=3)
    
    def create_input_section(self):
        input_container = tk.Frame(self.content_frame, bg=self.colors['bg_main'])
        input_container.pack(fill=tk.X, padx=40, pady=20)
        
        card_frame = tk.Frame(input_container, bg=self.colors['card_bg'], 
                             relief=tk.RAISED, bd=0)
        card_frame.pack(fill=tk.X, padx=10, pady=10)
        
        shadow = tk.Frame(input_container, bg='#0a1929')
        shadow.place(in_=card_frame, x=5, y=5, relwidth=1, relheight=1)
        card_frame.lift()
        
        card_title = tk.Label(card_frame,
                             text="Analysis Parameters",
                             font=('Helvetica', 16, 'bold'),
                             bg=self.colors['card_bg'],
                             fg=self.colors['success'])
        card_title.pack(pady=(15, 10))
        
        grid_frame = tk.Frame(card_frame, bg=self.colors['card_bg'])
        grid_frame.pack(padx=30, pady=(0, 20))
        
        self.create_input_field(grid_frame, "Stock TICKER SYMBOL", 0, "AAPL", 'symbol')
        self.create_dropdown_field(grid_frame, "Time Period", 1, 
                                   ['1mo', '3mo', '6mo', '1y', '2y', '5y'], '1y')
        self.create_input_field(grid_frame, "Predict Days", 2, "30", 'pred_days')
        
        self.create_gradient_button(card_frame)
    
    def create_input_field(self, parent, label_text, row, default_value, field_name):
        label = tk.Label(parent,
                        text=label_text,
                        font=('Helvetica', 12, 'bold'),
                        bg=self.colors['card_bg'],
                        fg=self.colors['text_primary'])
        label.grid(row=row, column=0, sticky='w', padx=20, pady=15)
        
        entry = tk.Entry(parent,
                        font=('Helvetica', 11),
                        bg=self.colors['bg_secondary'],
                        fg=self.colors['text_primary'],
                        insertbackground=self.colors['success'],
                        relief=tk.FLAT,
                        width=20,
                        bd=2)
        entry.insert(0, default_value)
        entry.grid(row=row, column=1, padx=20, pady=15, sticky='ew')
        
        entry.bind('<FocusIn>', 
                  lambda e: entry.config(highlightbackground=self.colors['accent_secondary'],
                                        highlightthickness=2))
        entry.bind('<FocusOut>', lambda e: entry.config(highlightthickness=0))
        
        if field_name == 'symbol':
            self.symbol_entry = entry
        elif field_name == 'pred_days':
            self.pred_days_entry = entry
    
    def create_dropdown_field(self, parent, label_text, row, values, default):
        label = tk.Label(parent,
                        text=label_text,
                        font=('Helvetica', 12, 'bold'),
                        bg=self.colors['card_bg'],
                        fg=self.colors['text_primary'])
        label.grid(row=row, column=0, sticky='w', padx=20, pady=15)
        
        self.period_var = tk.StringVar(value=default)
        dropdown = ttk.Combobox(parent,
                               textvariable=self.period_var,
                               values=values,
                               state='readonly',
                               font=('Helvetica', 11),
                               width=18,
                               style='Custom.TCombobox')
        dropdown.grid(row=row, column=1, padx=20, pady=15, sticky='ew')
    
    def create_gradient_button(self, parent):
        button_frame = tk.Frame(parent, bg=self.colors['card_bg'])
        button_frame.pack(pady=(10, 20))
        
        self.analyze_button = tk.Button(button_frame,
                                        text="ANALYZE STOCK",
                                        font=('Helvetica', 14, 'bold'),
                                        bg=self.colors['accent_primary'],
                                        fg=self.colors['text_primary'],
                                        activebackground=self.colors['button_gradient_2'],
                                        activeforeground=self.colors['text_primary'],
                                        relief=tk.FLAT,
                                        padx=40,
                                        pady=15,
                                        cursor='hand2',
                                        command=self.analyze_stock)
        self.analyze_button.pack()
        
        self.analyze_button.bind('<Enter>', 
                                lambda e: self.analyze_button.config(
                                    bg=self.colors['button_gradient_2']))
        self.analyze_button.bind('<Leave>', 
                                lambda e: self.analyze_button.config(
                                    bg=self.colors['accent_primary']))
    
    def create_info_cards(self):
        cards_container = tk.Frame(self.content_frame, bg=self.colors['bg_main'])
        cards_container.pack(fill=tk.X, padx=40, pady=20)
        
        self.info_cards = {}
        
        cards_data = [
            ("Company", "--", self.colors['accent_secondary']),
            ("Current Price", "$--", self.colors['success']),
            ("Change", "--", self.colors['warning']),
            ("Volume", "--", self.colors['accent_primary']),
            ("Predicted Price", "$--", self.colors['button_gradient_1']),
            ("Predicted Change", "--", self.colors['button_gradient_2'])
        ]
        
        for idx, (title, value, color) in enumerate(cards_data):
            row = idx // 3
            col = idx % 3
            
            card = tk.Frame(cards_container, bg=color, relief=tk.RAISED, bd=0)
            card.grid(row=row, column=col, padx=15, pady=15, sticky='nsew')
            cards_container.grid_columnconfigure(col, weight=1)
            
            title_label = tk.Label(card,
                                  text=title,
                                  font=('Helvetica', 10, 'bold'),
                                  bg=color,
                                  fg=self.colors['text_primary'])
            title_label.pack(pady=(15, 5))
            
            value_label = tk.Label(card,
                                  text=value,
                                  font=('Helvetica', 16, 'bold'),
                                  bg=color,
                                  fg=self.colors['text_primary'])
            value_label.pack(pady=(5, 15))
            
            self.info_cards[title] = (value_label, card, color)
            
            card.bind('<Enter>', 
                     lambda e, c=card, original=color: c.config(
                         bg=self._lighten_color(original)))
            card.bind('<Leave>', 
                     lambda e, c=card, original=color: c.config(bg=original))
    
    def _lighten_color(self, hex_color):
        return hex_color
    
    def create_chart_section(self):
        chart_container = tk.Frame(self.content_frame, bg=self.colors['bg_main'])
        chart_container.pack(fill=tk.BOTH, expand=True, padx=40, pady=20)
        
        chart_card = tk.Frame(chart_container, bg=self.colors['card_bg'], 
                             relief=tk.RAISED, bd=0)
        chart_card.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        chart_title = tk.Label(chart_card,
                              text="Interactive Price Chart",
                              font=('Helvetica', 16, 'bold'),
                              bg=self.colors['card_bg'],
                              fg=self.colors['success'])
        chart_title.pack(pady=(15, 10))
        
        plt.style.use('dark_background')
        self.figure = Figure(figsize=(14, 7), dpi=100, 
                            facecolor=self.colors['card_bg'])
        
        self.canvas = FigureCanvasTkAgg(self.figure, chart_card)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, 
                                         padx=15, pady=15)
    
    def fetch_stock_data(self, symbol, period):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if df.empty:
                raise ValueError("No data found for this symbol")
            
            info = stock.info
            company_name = info.get('longName', symbol)
            
            return df, company_name
        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")
    
    def predict_prices(self, df, days):
        df = df.dropna()
        df['Days'] = np.arange(len(df))
        
        X = df[['Days']].values
        y = df['Close'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        last_day = X[-1][0]
        future_days = np.array([[last_day + i] for i in range(1, days + 1)])
        predictions = model.predict(future_days)
        
        last_date = df.index[-1]
        pred_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=days, freq='D')
        
        r2 = model.score(X, y)
        
        return pred_dates, predictions, r2
    
    def update_info_display(self, company_name, current_price, change, 
                           volume, pred_price, pred_change):
        self.info_cards['Company'][0].config(text=company_name[:20])
        self.info_cards['Current Price'][0].config(text=f"${current_price:.2f}")
        
        change_percent = (change/current_price)*100
        change_text = f"{change:+.2f}\n({change_percent:+.2f}%)"
        self.info_cards['Change'][0].config(text=change_text)
        
        if change >= 0:
            self.info_cards['Change'][1].config(bg='#00f2a0')
        else:
            self.info_cards['Change'][1].config(bg='#ff6b6b')
        
        volume_text = f"{volume:,.0f}"
        if volume >= 1_000_000:
            volume_text = f"{volume/1_000_000:.2f}M"
        self.info_cards['Volume'][0].config(text=volume_text)
        
        self.info_cards['Predicted Price'][0].config(text=f"${pred_price:.2f}")
        
        pred_change_percent = (pred_change/current_price)*100
        pred_text = f"{pred_change:+.2f}\n({pred_change_percent:+.2f}%)"
        self.info_cards['Predicted Change'][0].config(text=pred_text)
        
        if pred_change >= 0:
            self.info_cards['Predicted Change'][1].config(bg='#4ecdc4')
        else:
            self.info_cards['Predicted Change'][1].config(bg='#ff6b9d')
    
    def plot_data(self, df, pred_dates, predictions):
        self.figure.clear()
        self.figure.patch.set_facecolor(self.colors['card_bg'])
        
        ax1 = self.figure.add_subplot(2, 1, 1, facecolor=self.colors['bg_secondary'])
        ax2 = self.figure.add_subplot(2, 1, 2, facecolor=self.colors['bg_secondary'])
        
        ax1.plot(df.index, df['Close'], label='Actual Price', 
                color='#00f2fe', linewidth=3, alpha=0.8)
        ax1.fill_between(df.index, df['Close'], alpha=0.3, color='#00f2fe')
        
        ax1.plot(pred_dates, predictions, label='Predicted Price', 
                color='#f093fb', linestyle='--', linewidth=3, 
                marker='o', markersize=4, alpha=0.9)
        ax1.fill_between(pred_dates, predictions, alpha=0.2, color='#f093fb')
        
        ax1.set_title('Stock Price Analysis & Forecast', 
                     fontsize=16, fontweight='bold', 
                     color=self.colors['success'], pad=20)
        ax1.set_xlabel('Date', fontsize=11, color=self.colors['text_secondary'])
        ax1.set_ylabel('Price ($)', fontsize=11, color=self.colors['text_secondary'])
        ax1.legend(loc='upper left', framealpha=0.9, fontsize=10)
        ax1.grid(True, alpha=0.2, linestyle='--', 
                color=self.colors['text_secondary'])
        ax1.tick_params(colors=self.colors['text_secondary'])
        
        for spine in ax1.spines.values():
            spine.set_color(self.colors['text_secondary'])
            spine.set_linewidth(0.5)
        
        colors_volume = ['#ffd93d' if v >= df['Volume'].mean() else '#6c5ce7' 
                        for v in df['Volume']]
        ax2.bar(df.index, df['Volume'], color=colors_volume, alpha=0.7, width=1)
        
        ax2.set_title('Trading Volume', fontsize=16, fontweight='bold', 
                     color=self.colors['warning'], pad=20)
        ax2.set_xlabel('Date', fontsize=11, color=self.colors['text_secondary'])
        ax2.set_ylabel('Volume', fontsize=11, color=self.colors['text_secondary'])
        ax2.grid(True, alpha=0.2, linestyle='--', 
                color=self.colors['text_secondary'])
        ax2.tick_params(colors=self.colors['text_secondary'])
        
        for spine in ax2.spines.values():
            spine.set_color(self.colors['text_secondary'])
            spine.set_linewidth(0.5)
        
        self.figure.tight_layout(pad=3)
        self.canvas.draw()
    
    def analyze_stock(self):
        try:
            symbol = self.symbol_entry.get().strip().upper()
            period = self.period_var.get()
            pred_days = int(self.pred_days_entry.get())
            
            if not symbol:
                self.show_colorful_message("Error", 
                                          "Please enter a stock symbol", "error")
                return
            
            if pred_days <= 0 or pred_days > 365:
                self.show_colorful_message("Error", 
                                          "Prediction days must be between 1 and 365", 
                                          "error")
                return
            
            self.analyze_button.config(text="ANALYZING...", 
                                      bg=self.colors['warning'])
            self.root.update()
            
            df, company_name = self.fetch_stock_data(symbol, period)
            pred_dates, predictions, r2_score = self.predict_prices(df, pred_days)
            
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            change = current_price - prev_price
            volume = df['Volume'].iloc[-1]
            
            pred_price = predictions[-1]
            pred_change = pred_price - current_price
            
            self.update_info_display(company_name, current_price, change, 
                                    volume, pred_price, pred_change)
            self.plot_data(df, pred_dates, predictions)
            
            confidence = 'High' if r2_score > 0.8 else 'Medium' if r2_score > 0.5 else 'Low'
            self.show_colorful_message("Success", 
                                      f"Analysis completed for {company_name}\n\n"
                                      f"Model Accuracy (R²): {r2_score:.4f}\n"
                                      f"Confidence: {confidence}", 
                                      "success")
        
        except Exception as e:
            self.show_colorful_message("Error", str(e), "error")
        
        finally:
            self.analyze_button.config(text="ANALYZE STOCK", 
                                      bg=self.colors['accent_primary'])
    
    def show_colorful_message(self, title, message, msg_type):
        if msg_type == "success":
            messagebox.showinfo(title, message)
        else:
            messagebox.showerror(title, message)

def main():
    root = tk.Tk()
    try:
        root.iconbitmap('icon.ico')
    except:
        pass
    app = StockPredictionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()