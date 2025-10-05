from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict
from collections import defaultdict


def load_data(file_path: str) -> pd.DataFrame:
    """Load financial data from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def returns_data(df: pd.DataFrame) -> pd.Series:
    """Preprocess the financial data."""
    df = df.sort_values(by="Date")
    df["Returns"] = df["Close"].pct_change()
    df = df.dropna()
    return df["Returns"]

def choose_same_length(files: List[Path]) -> Dict[str, int]:
    """Identify files with the same number of rows."""
    shapes = {x.stem: load_data(x).shape[0] for x in files}

    # value → [keys...]
    by_val = defaultdict(list)
    for k, v in shapes.items():
        by_val[v].append(k)

    groups = [keys for keys in by_val.values() if len(keys) > 1]
    for group in groups:
        print(group, len(group))


if __name__ == "__main__":
    # files = list((Path.cwd()/"data"/"raw"/"stocks").glob("*.csv"))
    # choose_same_length(files)

    # Create a DataFrame with returns of selected stocks
    symbols = ['CVNA', 'CLDR', 'XRF', 'CFBI', 'CTDD', 'ZYME', 'EEX', 'SVRA', 'NCSM']
    returns = {x: returns_data(load_data(Path.cwd()/"data"/"raw"/"stocks"/f"{x}.csv")) for x in symbols}
    df = pd.DataFrame(returns)

    # Create directory if not exists
    Path(Path.cwd()/"data"/"processed"/"stocks").mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(Path.cwd()/"data"/"processed"/"stocks"/"returns.csv", index=False)


# ['AMTBB', 'CMCTP', 'TZAC', 'OSMT', 'AMTB', 'GNAF', 'PHAS', 'DUKB'] 8
# ['CVNA', 'CLDR', 'XRF', 'CFBI', 'CTDD', 'ZYME', 'EEX', 'SVRA', 'NCSM'] 9
# ['BCLI', 'XPO', 'LMNR', 'DCAR', 'CRWS', 'CBFV', 'RUSHA', 'CAAS', 'HQI'] 9
# ['DAUD', 'UAUD', 'UEUR', 'UCHF', 'DCHF', 'DEUR', 'DGBP', 'UGBP', 'UJPY', 'DJPY'] 10
# ['ZYXI', 'HTD', 'TYG', 'WIW', 'SCD', 'MGLN', 'OGEN', 'IGR', 'UTG'] 9
# ['IAG', 'GENC', 'EFSC', 'CORV', 'APDN', 'HPI', 'GLNG', 'APWC'] 8
# ['PZC', 'PYN', 'BYM', 'BAF', 'BSE', 'PMX'] 6

