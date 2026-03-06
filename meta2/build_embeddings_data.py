# build_embeddings_from_tabela.py

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from embeddings_extractor import acc_segmentation, resample_to_30hz_5s, load_model
from ficheiros.constantes import COL_WITH_PARTICIPANT as COL
from ficheiros.initializar import importParaTabelaComParticipante

def build_embeddings_from_tabela(tabela):
    data = np.asarray(tabela)
    
    all_resampled_segments = []
    all_activities = []
    all_participants = []
    
    unique_participants = np.unique(data[:, COL["participante"]])
    for participant_id in unique_participants:
        participant_mask = data[:, COL["participante"]] == participant_id
        participant_data = data[participant_mask]
        
        # Remove participant column (column 0) before passing to acc_segmentation
        participant_data_no_part = participant_data[:, 1:]  # Keep columns 1 onwards

        original_segments, activities = acc_segmentation(participant_data_no_part)
        # Filter to keep only activities 1-7
        for segment, activity in zip(original_segments, activities):
            if 1 <= int(activity) <= 7:
                resampled_seg = resample_to_30hz_5s(segment, 51.5)[0]
                all_resampled_segments.append(resampled_seg)
                all_activities.append(int(activity))
                all_participants.append(int(participant_id))

    feature_encoder = load_model()
    embeddings_list = []

    resampled_segments = np.array(all_resampled_segments)
    print("Resampled segments shape:", resampled_segments.shape)

    x_all = np.transpose( np.array(resampled_segments), (0, 2, 1) )
    print(x_all.shape)

    # iterate over the resampled segments and pass them 
    #    through the model in batches to get the embeddings
    batch_size = 5
    with torch.no_grad():
        for i in range(0, x_all.shape[0], batch_size):
            xb = torch.from_numpy(x_all[i:i+batch_size]).float().to("cpu")
            eb = feature_encoder(xb)  # (B, D_embed)
            embeddings_list.append(eb.cpu().numpy())

    embeddings = np.concatenate(embeddings_list, axis=0)
    
    y = np.array(all_activities, dtype=int)
    participants = np.array(all_participants, dtype=int)

    return embeddings, y, participants


if __name__ == "__main__":

    tabela = importParaTabelaComParticipante(15, 5)

    embeddings, y, participants = build_embeddings_from_tabela(tabela)
    embeddings = embeddings.squeeze(-1) #remove a dimensão extra

    print("embeddings:", embeddings.shape, " | y:", y.shape, " | participants:", participants.shape)

    np.save("meta2/embeddings.npy", embeddings)
    np.save("meta2/y.npy", y)
    np.save("meta2/participants.npy", participants)
