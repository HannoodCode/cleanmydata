import pandas as pd
import sys, threading, time
import itertools
import os

class Spinner:
    """Terminal spinner for indicating progress."""

    def __init__(self, message="Processing...", delay=0.1):
        self.spinner = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        self.delay = delay
        self.message = message
        self.stop_running = False
        self.thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        while not self.stop_running:
            sys.stdout.write(f"\r{self.message} {next(self.spinner)}")
            sys.stdout.flush()
            time.sleep(self.delay)
        sys.stdout.write(f"\r{self.message} \n")
        sys.stdout.flush()

    def start(self):
        self.stop_running = False
        self.thread.start()

    def stop(self):
        self.stop_running = True
        self.thread.join()

def load_data(filepath, verbose=True):
    """Load CSV or Excel dataset with feedback and error handling."""
    from src.utils import Spinner

    spinner = None
    message = f"Loading dataset: {os.path.basename(filepath)}"
    if not verbose:
        spinner = Spinner(message)
        spinner.start()
    else:
        print(f"\n{message} ...")

    try:
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filepath.endswith((".xls", ".xlsx")):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Supported: .csv, .xls, .xlsx")

        if df.empty:
            print("\nLoaded file is empty.")
            return pd.DataFrame()

        if spinner:
            spinner.stop()

        print(f"Successfully loaded {df.shape[0]:,} rows × {df.shape[1]} columns.\n")
        return df

    except FileNotFoundError:
        print(f"\nError: File not found at {filepath}")
    except pd.errors.EmptyDataError:
        print("\nError: The file is empty or invalid.")
    except pd.errors.ParserError:
        print("\nError: Parsing error occurred while reading the file.")
    except Exception as e:
        print(f"\nUnexpected error while loading file: {e}")
    finally:
        if spinner:
            spinner.stop()

    return pd.DataFrame()