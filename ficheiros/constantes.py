#---colunas-
COL = {
    "dispositivo": 0,
    "ax": 1, "ay": 2, "az": 3,
    "gx": 4, "gy": 5, "gz": 6,
    "mx": 7, "my": 8, "mz": 9,
    "tempo": 10,
    "atividade": 11,
}

# Column mapping when participant column is included (used by importParaTabelaComParticipante)
COL_WITH_PARTICIPANT = {
    "participante": 0,
    "dispositivo": 1,
    "ax": 2, "ay": 3, "az": 4,
    "gx": 5, "gy": 6, "gz": 7,
    "mx": 8, "my": 9, "mz": 10,
    "tempo": 11,
    "atividade": 12,
}

ACTIVITY_NAMES = {
    1: "Stand",
    2: "Sit",
    3: "Sit and Talk",
    4: "Walk",
    5: "Walk and Talk",
    6: "Climb Stair (up/down)",
    7: "Climb Stair (up/down) and talk",
    8: "Stand -> Sit",
    9: "Sit -> Stand",
    10: "Stand -> Sit and talk",
    11: "Sit -> Stand and talk",
    12: "Stand -> Walk",
    13: "Walk -> Stand",
    14: "Stand -> Climb stairs (up/down), Stand -> Climb stairs (up/down) and talk",
    15: "Climb stairs (up/down) -> Walk",
    16: "Climb stairs (up/down) and talk -> Walk and talk",
}

nomes_sensores = [
     'acc_x', 'acc_y', 'acc_z',      
     'gyro_x', 'gyro_y', 'gyro_z',   
     'mag_x', 'mag_y', 'mag_z'       
 ]

nomes_funcoes = [
    'mean',
    'median',            
    'std',               
    'var',        
    'rms',               
    'avg_deriv',         
    'skew',              
    'kurt',              
    'iqr',               
    'zcr',               
    'mcr',               
    'spectral_entropy'   
]