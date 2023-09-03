from pathlib import Path
import os


curr_file = Path(os.path.dirname(__file__))
data_dir = curr_file/'..'/'db'
regression_path = data_dir / 'synthetic'/'regression'
tick_base_dir = curr_file/'..'/"tick_data"

tickers = ['ES', 'NQ', 'CL', 'GL', 'TU', 'GC', 'CN']

exponents = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
