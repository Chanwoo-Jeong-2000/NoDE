# [WWW '25] GCNs Meets Long-tail: Negative Effects from Embedding Norm of GCN-Based Recommendations

# Requirements
python 3.8.18, cuda 11.8, and the following installations:
```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-geometric
pip install six
pip install pandas
```

# Run
Instead of **[dataset]**, substitute **Amazon-CD**, **Gowalla**, **Yelp** to run the code.
##### NoDE-LightGCN (Ours)
```
python main_NoDE-LightGCN.py --dataset [dataset]
```
##### LightGCN
```
python main_LightGCN.py --dataset [dataset]
```

# Results
The results of the experiments are in the **results4comparison** folder.
