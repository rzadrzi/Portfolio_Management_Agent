import os
import sys

from pathlib import Path

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, 'src')
PORTFOLIO = os.path.join(SRC, 'portfolio')

sys.path.append(SRC)
# sys.path.append(PORTFOLIO)

if __name__=="__main__":
    print(sys.path)