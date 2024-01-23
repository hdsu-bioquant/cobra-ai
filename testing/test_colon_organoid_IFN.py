from scontovae_train import *



#  Arguments
res_dir = "./projects/scontovae/data/model_colon_regressout_split0.6"
resolution = 300
fig_extension = "png"
############


ontobj = load_ontobj("./projects/scontovae/data/pathway/GO.ontobj")
adata = preprocess("./projects/integration_preprocessed.h5ad") # An h5ad file Converted from a Seurat .RDS file 

# Train/test split
train_adata, test_adata = split_adata(adata, 0.6)

# Integrate adata and ontobj
train_adata = integration(train_adata, ontobj)
test_adata = integration(test_adata, ontobj)

# Train the model
trainer(train_adata, res_dir,epochs=200)

# Load the model
model = loader(test_adata, res_dir)

# Get the codings on the UMAP
embedding, df_adata_compressed_2D = get_umap_embedding(test_adata, model)
df_adata_compressed_2D.plot(kind="scatter", x="UMAP_1", y="UMAP_2", c="modality", cmap="tab10", s = 5, figsize=(10,7))
save_fig(res_dir, "embedding_umap_tx", fig_extension, resolution)

df_adata_compressed_2D.plot(kind="scatter", x="UMAP_1", y="UMAP_2", alpha=0.8, c="cell_type", s=5, cmap='tab10', figsize=(10,7))
save_fig(res_dir, "cell_type", fig_extension, resolution)

# Get the predicted pathway activities
file_path = os.path.join(res_dir, "PW_activity_colontestdata_ileummodel_regressout_split0.6.csv")
df_act = get_pw_act(model, test_adata, file_path)



