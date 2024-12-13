import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import ImageGrab, Image, ImageTk
import pytesseract
import pandas as pd
import cv2
import numpy as np
import os
from datetime import datetime
import csv
import matplotlib
matplotlib.use('Agg')  # Use Agg backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
import re
import sqlite3

class FinanceScanner:
    def __init__(self, root):
        self.root = root
        self.root.title("Finance Statement Scanner")
        self.root.geometry("1200x800")
        
        # Initialize database
        self.init_database()
        
        # Initialize account data
        self.accounts = {
            "Lost Debit Card": None,
            "Current Credit Card": None,
            "Other Account": None
        }
        
        # Create main frame with notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create scanner frame
        self.scanner_frame = ttk.Frame(self.notebook, padding="10")
        self.scanner_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create viewer frame
        self.viewer_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viewer_frame, text="Balance History")
        
        # Create budget frame
        self.budget_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.budget_frame, text="Budget")
        
        # Add frames to notebook
        self.notebook.add(self.scanner_frame, text="Scanner")
        
        # Set up scanner interface
        self.setup_scanner()
        
        # Set up viewer interface
        self.setup_viewer()
        
        # Set up budget interface
        self.setup_budget()
        
        self.current_image = None

    def setup_scanner(self):
        # Create buttons frame
        button_frame = ttk.Frame(self.scanner_frame)
        button_frame.grid(row=0, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Take Screenshot", command=self.take_screenshot).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Process Image", command=self.process_image).pack(side=tk.LEFT, padx=5)
        
        # Create account balance frame
        balance_frame = ttk.LabelFrame(self.scanner_frame, text="Account Balances", padding="10")
        balance_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Create entry fields for each account
        self.balance_vars = {}
        for i, account in enumerate(self.accounts.keys()):
            ttk.Label(balance_frame, text=account).grid(row=i, column=0, sticky=tk.W, pady=5)
            var = tk.StringVar()
            entry = ttk.Entry(balance_frame, textvariable=var, width=20)
            entry.grid(row=i, column=1, padx=10, pady=5)
            self.balance_vars[account] = var
        
        # Create verification frame
        verify_frame = ttk.Frame(self.scanner_frame)
        verify_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(verify_frame, text="Verify & Save", command=self.verify_and_save).pack(side=tk.LEFT, padx=5)
        ttk.Button(verify_frame, text="Clear", command=self.clear_entries).pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.scanner_frame, textvariable=self.status_var).grid(row=3, column=0, columnspan=2, pady=10)

    def setup_viewer(self):
        # Create notebook for different views
        viewer_notebook = ttk.Notebook(self.viewer_frame)
        viewer_notebook.pack(fill='both', expand=True)
        
        # Create frames for different views
        self.table_frame = ttk.Frame(viewer_notebook)
        self.graph_frame = ttk.Frame(viewer_notebook)
        self.summary_frame = ttk.Frame(viewer_notebook)
        
        viewer_notebook.add(self.table_frame, text="Table View")
        viewer_notebook.add(self.graph_frame, text="Graph View")
        viewer_notebook.add(self.summary_frame, text="Summary")
        
        # Add refresh button at the top
        ttk.Button(self.viewer_frame, text="Refresh Data", command=self.refresh_data).pack(pady=(0, 5))
        
        # Initialize the views
        self.setup_table_view()
        self.setup_graph_view()
        self.setup_summary_view()
        
        # Load initial data
        self.refresh_data()

    def setup_table_view(self):
        # Create treeview
        columns = ("Timestamp", "Lost Debit Card", "Current Credit Card", "Other Account")
        self.tree = ttk.Treeview(self.table_frame, columns=columns, show="headings")
        
        # Set column headings
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150)
        
        # Add scrollbars
        y_scroll = ttk.Scrollbar(self.table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        x_scroll = ttk.Scrollbar(self.table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        
        # Grid layout
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        y_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        x_scroll.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.table_frame.columnconfigure(0, weight=1)
        self.table_frame.rowconfigure(0, weight=1)

    def setup_graph_view(self):
        # Create graph frame with controls
        self.graph_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.graph_frame, text="Graph View")

        # Add date range controls
        controls_frame = ttk.Frame(self.graph_frame)
        controls_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(controls_frame, text="Date Range:").pack(side='left', padx=5)
        self.date_range = ttk.Combobox(controls_frame, values=['1 Week', '1 Month', '3 Months', '6 Months', '1 Year', 'All'])
        self.date_range.set('1 Month')
        self.date_range.pack(side='left', padx=5)
        self.date_range.bind('<<ComboboxSelected>>', lambda e: self.refresh_data())

        # Create figure and axis
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasAgg(self.fig)
        
        # Create label to display the graph
        self.graph_label = tk.Label(self.graph_frame)
        self.graph_label.pack(fill=tk.BOTH, expand=True)
        
        # Initial render
        self.update_graph()

    def setup_summary_view(self):
        # Create summary labels
        self.summary_labels = {}
        accounts = ["Lost Debit Card", "Current Credit Card", "Other Account"]
        
        for i, account in enumerate(accounts):
            frame = ttk.LabelFrame(self.summary_frame, text=account, padding="10")
            frame.grid(row=i, column=0, sticky=(tk.W, tk.E), padx=10, pady=5)
            
            # Current Balance
            ttk.Label(frame, text="Current Balance:").grid(row=0, column=0, sticky=tk.W, pady=2)
            current_bal = ttk.Label(frame, text="$0.00")
            current_bal.grid(row=0, column=1, sticky=tk.W, padx=10, pady=2)
            
            # Change since last
            ttk.Label(frame, text="Change since last:").grid(row=1, column=0, sticky=tk.W, pady=2)
            change = ttk.Label(frame, text="$0.00")
            change.grid(row=1, column=1, sticky=tk.W, padx=10, pady=2)
            
            self.summary_labels[account] = {
                'current': current_bal,
                'change': change
            }

    def setup_budget(self):
        # Create top frame for controls
        control_frame = ttk.Frame(self.budget_frame)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # Year selector
        ttk.Label(control_frame, text="Year:").pack(side='left', padx=5)
        self.year_var = tk.StringVar(value=str(datetime.now().year))
        year_entry = ttk.Entry(control_frame, textvariable=self.year_var, width=6)
        year_entry.pack(side='left', padx=5)
        
        # Add category button
        add_cat_btn = ttk.Button(control_frame, text="Add Category", command=self.add_budget_category)
        add_cat_btn.pack(side='left', padx=5)
        
        # Create treeview for budget
        self.budget_tree = ttk.Treeview(self.budget_frame, show='headings')
        
        # Configure columns
        columns = ['Category'] + [f'{m[:3]}' for m in ['January', 'February', 'March', 'April', 'May', 'June', 
                                                      'July', 'August', 'September', 'October', 'November', 'December']] + ['Total']
        self.budget_tree['columns'] = columns
        
        # Configure column headings
        for col in columns:
            self.budget_tree.heading(col, text=col)
            width = 100 if col == 'Category' else 70
            self.budget_tree.column(col, width=width, anchor='e')
        
        # Add scrollbars
        y_scroll = ttk.Scrollbar(self.budget_frame, orient='vertical', command=self.budget_tree.yview)
        x_scroll = ttk.Scrollbar(self.budget_frame, orient='horizontal', command=self.budget_tree.xview)
        self.budget_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        
        # Pack everything
        self.budget_tree.pack(fill='both', expand=True)
        y_scroll.pack(side='right', fill='y')
        x_scroll.pack(side='bottom', fill='x')
        
        # Bind double-click event for editing
        self.budget_tree.bind('<Double-1>', self.edit_budget_item)
        
        # Load saved budget data if exists
        self.load_budget_data()

    def init_database(self):
        # Create database connection
        self.conn = sqlite3.connect('finance_scanner.db')
        self.cursor = self.conn.cursor()
        
        # Create budget table if it doesn't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS budget_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT NOT NULL
            )
        ''')
        
        # Create budget amounts table if it doesn't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS budget_amounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_id INTEGER,
                year INTEGER,
                month INTEGER,
                amount REAL,
                FOREIGN KEY (category_id) REFERENCES budget_categories (id),
                UNIQUE(category_id, year, month)
            )
        ''')
        
        self.conn.commit()

    def add_budget_category(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Budget Category")
        dialog.geometry("300x100")
        
        ttk.Label(dialog, text="Category Name:").pack(padx=5, pady=5)
        entry = ttk.Entry(dialog)
        entry.pack(padx=5, pady=5)
        
        def add():
            category = entry.get().strip()
            if category:
                # Check if category already exists
                self.cursor.execute('SELECT id FROM budget_categories WHERE category_name = ?', (category,))
                if self.cursor.fetchone():
                    messagebox.showerror("Error", "Category already exists")
                    return
                
                values = [category] + ['0.00'] * 12 + ['0.00']
                self.budget_tree.insert('', 'end', values=values)
                self.calculate_totals()
                self.save_budget_data()
                dialog.destroy()
        
        ttk.Button(dialog, text="Add", command=add).pack(pady=10)

    def edit_budget_item(self, event):
        item = self.budget_tree.selection()[0]
        column = self.budget_tree.identify_column(event.x)
        col_num = int(column[1]) - 1
        
        if col_num == 0:  # Category name
            return
        
        if col_num == 13:  # Total column
            return
            
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Amount")
        dialog.geometry("200x100")
        
        current_value = self.budget_tree.item(item)['values'][col_num]
        
        ttk.Label(dialog, text="Amount:").pack(padx=5, pady=5)
        entry = ttk.Entry(dialog)
        entry.insert(0, current_value)
        entry.pack(padx=5, pady=5)
        
        def update():
            try:
                new_value = float(entry.get())
                values = list(self.budget_tree.item(item)['values'])
                values[col_num] = f"{new_value:.2f}"
                self.budget_tree.item(item, values=values)
                self.calculate_totals()
                self.save_budget_data()
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number")
        
        ttk.Button(dialog, text="Update", command=update).pack(pady=10)

    def calculate_totals(self):
        for item in self.budget_tree.get_children():
            values = list(self.budget_tree.item(item)['values'])
            total = sum(float(x) for x in values[1:13])  # Sum all months
            values[13] = f"{total:.2f}"
            self.budget_tree.item(item, values=values)

    def save_budget_data(self):
        try:
            for item in self.budget_tree.get_children():
                values = self.budget_tree.item(item)['values']
                category_name = values[0]
                year = int(self.year_var.get())
                
                # Get or create category
                self.cursor.execute('SELECT id FROM budget_categories WHERE category_name = ?', (category_name,))
                result = self.cursor.fetchone()
                
                if result:
                    category_id = result[0]
                else:
                    self.cursor.execute('INSERT INTO budget_categories (category_name) VALUES (?)', (category_name,))
                    category_id = self.cursor.lastrowid
                
                # Save monthly amounts
                for month in range(12):
                    amount = float(values[month + 1])
                    self.cursor.execute('''
                        INSERT OR REPLACE INTO budget_amounts (category_id, year, month, amount)
                        VALUES (?, ?, ?, ?)
                    ''', (category_id, year, month + 1, amount))
            
            self.conn.commit()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save budget data: {str(e)}")
    
    def load_budget_data(self):
        try:
            # Clear existing items
            for item in self.budget_tree.get_children():
                self.budget_tree.delete(item)
            
            year = int(self.year_var.get())
            
            # Get all categories and their amounts
            self.cursor.execute('''
                SELECT 
                    c.category_name,
                    GROUP_CONCAT(a.amount) as amounts
                FROM budget_categories c
                LEFT JOIN budget_amounts a ON c.id = a.category_id 
                    AND a.year = ?
                GROUP BY c.id, c.category_name
                ORDER BY c.category_name
            ''', (year,))
            
            results = self.cursor.fetchall()
            
            for category_name, amounts in results:
                if amounts:
                    amounts = [float(x) if x else 0.0 for x in amounts.split(',')]
                else:
                    amounts = [0.0] * 12
                    
                # Calculate total
                total = sum(amounts)
                
                # Format values
                values = [category_name] + [f"{amount:.2f}" for amount in amounts] + [f"{total:.2f}"]
                self.budget_tree.insert('', 'end', values=values)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load budget data: {str(e)}")

    def refresh_data(self):
        try:
            # Load data from CSV
            csv_path = os.path.join(os.path.expanduser("~/Downloads"), "account_balances.csv")
            if not os.path.exists(csv_path):
                return
            
            df = pd.read_csv(csv_path)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            # Update table view
            self.update_table(df)
            
            # Update graph view
            self.update_graph(df)
            
            # Update summary view
            self.update_summary(df)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")

    def update_table(self, df):
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add new items
        for _, row in df.iterrows():
            values = (
                row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                f"${row['Lost Debit Card']:,.2f}",
                f"${row['Current Credit Card']:,.2f}",
                f"${row['Other Account']:,.2f}"
            )
            self.tree.insert('', 'end', values=values)

    def update_graph(self, df=None):
        if not hasattr(self, 'canvas'):
            return
            
        if df is not None and not df.empty:
            self.ax.clear()
            
            # Filter data based on date range
            if hasattr(self, 'date_range'):
                range_text = self.date_range.get()
                if range_text != 'All':
                    # Parse the range
                    number = int(range_text.split()[0])
                    unit = range_text.split()[1].lower()
                    
                    # Calculate the start date
                    if unit == 'week':
                        start_date = pd.Timestamp.now() - pd.Timedelta(weeks=number)
                    elif unit == 'month':
                        start_date = pd.Timestamp.now() - pd.DateOffset(months=number)
                    elif unit == 'year':
                        start_date = pd.Timestamp.now() - pd.DateOffset(years=number)
                    
                    # Filter the dataframe
                    df = df[df['Timestamp'] >= start_date]
            
            # Plot each account
            colors = {'Lost Debit Card': '#2ecc71', 'Current Credit Card': '#e74c3c', 'Other Account': '#3498db'}
            for column, color in colors.items():
                self.ax.plot(df['Timestamp'], df[column], label=column, marker='o', color=color)
            
            # Customize the graph
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Balance ($)')
            self.ax.set_title('Account Balances Over Time')
            self.ax.legend()
            self.ax.grid(True, linestyle='--', alpha=0.7)
            
            # Format dates on x-axis
            self.fig.autofmt_xdate()  # Rotate and align the tick labels
            
            # Use dollar format for y-axis
            self.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.2f}'))
            
            # Add gridlines
            self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Draw the figure on the canvas
        self.canvas.draw()
        
        # Convert canvas to image
        buf = io.BytesIO()
        self.canvas.print_png(buf)
        buf.seek(0)
        
        # Convert to PhotoImage
        img = Image.open(buf)
        self.graph_image = ImageTk.PhotoImage(img)
        
        # Update label
        self.graph_label.config(image=self.graph_image)
        self.graph_label.image = self.graph_image
        
        # Close buffer
        buf.close()

    def update_summary(self, df):
        if len(df) < 1:
            return
            
        # Get latest and previous records
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        
        # Update summary for each account
        for account in ['Lost Debit Card', 'Current Credit Card', 'Other Account']:
            current_balance = latest[account]
            change = current_balance - previous[account] if len(df) > 1 else 0
            
            # Update labels
            self.summary_labels[account]['current'].config(
                text=f"${current_balance:,.2f}"
            )
            self.summary_labels[account]['change'].config(
                text=f"${change:,.2f}",
                foreground='green' if change >= 0 else 'red'
            )

    def take_screenshot(self):
        # Don't minimize the main window
        self._create_selection_window()

    def _create_selection_window(self):
        # Create a transparent overlay window
        self.selection_window = tk.Toplevel(self.root)
        
        # Set up window properties for macOS
        self.selection_window.attributes('-alpha', 0.3)
        self.selection_window.attributes('-topmost', True)
        self.selection_window.overrideredirect(True)
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Position the window to cover the full screen
        self.selection_window.geometry(f"{screen_width}x{screen_height}+0+0")
        
        # Create canvas with white background
        self.canvas = tk.Canvas(
            self.selection_window,
            highlightthickness=0,
            bg='white'
        )
        self.canvas.pack(fill='both', expand=True)
        
        # Make the window and canvas transparent
        self.selection_window.wait_visibility(self.selection_window)
        self.selection_window.wm_attributes('-alpha', 0.3)
        
        # Configure cursor
        self.canvas.configure(cursor="crosshair")
        
        # Bind mouse events to canvas
        self.canvas.bind('<Button-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)
        
        # Bind escape key to the selection window
        self.selection_window.bind('<Escape>', lambda e: self._cancel_screenshot())
        
        # Add instruction label
        instruction_label = tk.Label(
            self.selection_window,
            text="Click and drag to select area. Press ESC to cancel.",
            bg='white',
            fg='black',
            pady=5
        )
        instruction_label.pack(side='top', fill='x')
        
        # Initialize selection variables
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.dim_text = None

    def _on_mouse_down(self, event):
        # Clear any existing rectangles
        if self.rect:
            self.canvas.delete(self.rect)
        if self.dim_text:
            self.canvas.delete(self.dim_text)
            
        # Store starting position
        self.start_x = event.x
        self.start_y = event.y
        
        # Create new rectangle
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline='red',
            width=2
        )

    def _on_mouse_drag(self, event):
        if not self.rect:
            return
            
        # Update rectangle size
        self.canvas.coords(
            self.rect,
            self.start_x, self.start_y,
            event.x, event.y
        )
        
        # Update dimension display
        width = abs(event.x - self.start_x)
        height = abs(event.y - self.start_y)
        
        # Delete old dimension text
        if self.dim_text:
            self.canvas.delete(self.dim_text)
            
        # Create new dimension text
        self.dim_text = self.canvas.create_text(
            min(self.start_x, event.x),
            min(self.start_y, event.y) - 10,
            text=f'{width} x {height}',
            fill='red',
            anchor='sw'
        )

    def _on_mouse_up(self, event):
        if not self.rect:
            return
            
        # Get the coordinates
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)
        
        # Ensure minimum selection size
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            # Hide the selection window
            self.selection_window.withdraw()
            
            # Schedule the screenshot capture
            self.root.after(100, lambda: self._do_capture_screenshot((x1, y1, x2, y2)))
        else:
            self._cancel_screenshot()

    def _do_capture_screenshot(self, bbox):
        try:
            # Get Downloads folder path
            downloads_path = os.path.expanduser("~/Downloads")
            
            # Capture the screenshot
            screenshot = ImageGrab.grab(bbox=bbox)
            
            # Generate filename and save to Downloads
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(downloads_path, f"screenshot_{timestamp}.png")
            
            screenshot = screenshot.convert('RGB')
            screenshot.save(filename, "PNG")
            
            self.current_image = filename
            self.status_var.set(f"Screenshot saved to Downloads: {os.path.basename(filename)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture screenshot: {str(e)}")
            print(traceback.format_exc())
        finally:
            self._cancel_screenshot()

    def _cancel_screenshot(self):
        if hasattr(self, 'dim_text') and self.dim_text:
            self.canvas.delete(self.dim_text)
        if hasattr(self, 'rect') and self.rect:
            self.canvas.delete(self.rect)
        if self.selection_window:
            self.selection_window.destroy()
            self.selection_window = None
        self.root.deiconify()

    def load_image(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if filename:
            self.current_image = filename
            self.status_var.set(f"Loaded image: {filename}")

    def process_image(self):
        if not self.current_image:
            messagebox.showwarning("Warning", "Please take a screenshot or load an image first")
            return

        try:
            # Read image
            image = cv2.imread(self.current_image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding to preprocess the image
            threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # Perform OCR
            text = pytesseract.image_to_string(threshold)
            
            # Parse the OCR text to find account balances
            balances = self._parse_financial_text(text)
            
            # Update the entry fields with the found balances
            for account, amount in balances.items():
                if amount:
                    self.balance_vars[account].set(amount)
            
            self.status_var.set("Image processed. Please verify and save the balances.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")

    def _parse_financial_text(self, text):
        """Parse the OCR text to find account balances"""
        # Initialize results
        found_balances = {
            "Lost Debit Card": None,
            "Current Credit Card": None,
            "Other Account": None
        }
        
        # Convert text to lowercase for easier matching
        text_lower = text.lower()
        lines = text_lower.split('\n')
        
        # Account keywords with fuzzy matching
        account_keywords = {
            "Lost Debit Card": ['debit', 'lost', 'checking'],
            "Current Credit Card": ['credit', 'current', 'card'],
            "Other Account": ['other', 'savings', 'additional']
        }
        
        # Look for balance amounts in each line
        for line in lines:
            # Try to find currency amounts using regex
            amounts = re.findall(r'-?\$?\s*\d+(?:,\d{3})*(?:\.\d{2})?', line)
            
            if amounts:
                # Get the last amount found in the line
                amount = amounts[-1]
                # Clean the amount string
                clean_amount = amount.replace('$', '').replace(',', '').strip()
                
                # Check each account's keywords
                for account, keywords in account_keywords.items():
                    if any(keyword in line for keyword in keywords):
                        found_balances[account] = clean_amount
                        break
        
        return found_balances

    def _is_currency(self, text):
        """Check if text looks like a currency amount"""
        # Regex pattern for currency: optional negative sign, optional $, digits with optional commas, optional decimal part
        currency_pattern = r'^-?\$?\s*\d+(?:,\d{3})*(?:\.\d{2})?$'
        return bool(re.match(currency_pattern, text.strip()))

    def verify_and_save(self):
        """Verify and save the account balances"""
        try:
            # Collect current values
            balances = {}
            for account, var in self.balance_vars.items():
                value = var.get().strip()
                if not value:
                    messagebox.showwarning("Warning", f"Please enter a value for {account}")
                    return
                try:
                    # Remove currency symbols and separators
                    clean_value = value.replace('$', '').replace(',', '')
                    balances[account] = float(clean_value)
                except ValueError:
                    messagebox.showerror("Error", f"Invalid amount for {account}: {value}")
                    return
            
            # Ask for confirmation
            confirm_msg = "Please verify these balances:\n\n"
            for account, amount in balances.items():
                confirm_msg += f"{account}: ${amount:,.2f}\n"
            confirm_msg += "\nAre these correct?"
            
            if messagebox.askyesno("Confirm Balances", confirm_msg):
                # Save to CSV
                self._save_balances(balances)
                self.clear_entries()
                self.refresh_data()  # Refresh the viewer after saving
                messagebox.showinfo("Success", "Balances have been saved!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save balances: {str(e)}")

    def _save_balances(self, balances):
        """Save balances to CSV file"""
        # Get Downloads folder path
        downloads_path = os.path.expanduser("~/Downloads")
        csv_path = os.path.join(downloads_path, "account_balances.csv")
        
        # Prepare data for CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row_data = {
            'Timestamp': timestamp,
            **balances
        }
        
        # Check if file exists
        file_exists = os.path.exists(csv_path)
        
        # Write to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Timestamp'] + list(self.accounts.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)

    def clear_entries(self):
        """Clear all entry fields"""
        for var in self.balance_vars.values():
            var.set("")

def main():
    root = tk.Tk()
    app = FinanceScanner(root)
    root.mainloop()

if __name__ == "__main__":
    main()
