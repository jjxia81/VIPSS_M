import numpy as np
import math

def cal_memory(npt):
    n_rows = 4 * npt + 4
    m = n_rows * n_rows * 8
    return m

rows = [500, 1000, 2000, 4000, 6000, 8000, 10000]
for row in rows:
    m_all = cal_memory(row)
    print(row, m_all) 
    m_g = m_all / (1024 * 1024 * 1024) 
    print(row, m_g) 
    print(row, m_g * 3) 