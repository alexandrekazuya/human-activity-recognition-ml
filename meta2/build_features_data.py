import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from mod4_2 import criarJanelasComParticipantes, extrairFeaturesJanela
from ficheiros.initializar import importParaTabelaComParticipante


def build_features_from_tabela(participantes=15, devices=5):
    tabela = importParaTabelaComParticipante(participantes, devices)

    # criar janelas com participantes
    janelas, atividades, participantes_arr = criarJanelasComParticipantes(tabela)

    print(f"Created {len(janelas)} windows from table (participants returned: {len(participantes_arr)})")

    features_list = []
    kept_ativ = []
    kept_parts = []

    # extract features per window, but keep only activities 1..7 (consistent with embeddings builder)
    for janela, ativ, part in zip(janelas, atividades, participantes_arr):
        a = int(ativ)
        if 1 <= a <= 7:
            feats = extrairFeaturesJanela(janela)
            features_list.append(feats)
            kept_ativ.append(a)
            kept_parts.append(int(part))

    X = np.array(features_list)
    y = np.array(kept_ativ, dtype=int)
    parts = np.array(kept_parts, dtype=int)

    print("features shape:", X.shape, "y:", y.shape, "participants:", parts.shape)

    print("participantes unique: ", np.unique(parts))

    np.save("meta2/features.npy", X)
    np.save("meta2/y_features.npy", y)
    np.save("meta2/participants_features.npy", parts)

    return X, y, parts


if __name__ == '__main__':
    build_features_from_tabela()
