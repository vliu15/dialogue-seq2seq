python3 train.py -data data/iac/train.data.pt -save_model trained \
  -save_mode best -proj_share_weight -label_smoothing -embs_share_weight
  # -src_emb_file data/glove/src_emb_file.npy -tgt_emb_file data/glove/tgt_emb_file.npy

