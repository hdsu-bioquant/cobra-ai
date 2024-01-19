from scontovae_train import *

ontobj = load_ontobj("./projects/scontovae/data/pathway/GO.ontobj")
adata = preprocess("./projects/integration_preprocessed.h5ad") # Converted from Seurat .RDS file 

# Train/test split
train_adata, test_adata = split_adata(adata, 0.6)

train_adata = integration(train_adata, ontobj)
test_adata = integration(test_adata, ontobj)

trainer(train_adata, "./projects/scontovae/data/model_colon_random_0.6_regressout",epochs=100)


model = loader(test_adata, "./projects/scontovae/data/model_colon_random_0.6_regressout")

embedding, df_adata_compressed_2D = get_umap_embedding(test_adata, model)

df_act = get_pw_act(model, test_adata, "./projects/scontovae/results/PW_activity_Colon_model_random_0.6_regress.csv")

df_adata_compressed_2D.plot(kind="scatter", x="UMAP_1", y="UMAP_2", c="modality", cmap="tab10", s = 5, figsize=(10,7))
df_adata_compressed_2D.plot(kind="scatter", x="UMAP_1", y="UMAP_2", alpha=0.8, c="cell_type", s=5, cmap='tab10', figsize=(10,7))

