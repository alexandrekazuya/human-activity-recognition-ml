import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np

#--- extrair dados---
def importParaTabela(participantes, devices):
    for p in range(0,participantes):
        for d in range(1, devices+1):
            path = f'./data/FORTH_TRACE_DATASET-master/part{p}/part{p}dev{d}.csv'
            if (p==0 and d==1):
                tabelaGigante = np.genfromtxt(path, delimiter=',')
                continue
            tabelaGigante = np.vstack((tabelaGigante, np.genfromtxt(path, delimiter=',')))
    return tabelaGigante


def importParaTabelaComParticipante(participantes, devices):
    """Same as importParaTabela but adds participant ID as first column"""
    primeira = True
    for p in range(0,participantes):
        for d in range(1, devices+1):
            path = f'./data/FORTH_TRACE_DATASET-master/part{p}/part{p}dev{d}.csv'
            try:
                dados = np.genfromtxt(path, delimiter=',')
                if dados.size == 0:
                    print(f"WARNING: Empty file {path}")
                    continue
            except Exception as e:
                print(f"WARNING: Failed to load {path}: {e}")
                continue
            
            # Add participant column as first column
            part_col = np.full((dados.shape[0], 1), p)
            dados_com_part = np.hstack((part_col, dados))
            
            if primeira:
                tabelaGigante = dados_com_part
                primeira = False
            else:
                tabelaGigante = np.vstack((tabelaGigante, dados_com_part))
    
    #print(f"DEBUG importParaTabelaComParticipante: Loaded {tabelaGigante.shape[0]} rows, unique participants: {np.unique(tabelaGigante[:, 0])}")
    return tabelaGigante


def groupByDeviceActivity(tabela, COL):
    dev = tabela[:, COL["dispositivo"]].astype(int)
    act = tabela[:, COL["atividade"]].astype(int)
    devices = np.unique(dev)
    activities = np.unique(act)
    grid = {d: {a: tabela[(dev == d) & (act == a)] for a in activities} for d in devices}
    return devices, activities, grid
