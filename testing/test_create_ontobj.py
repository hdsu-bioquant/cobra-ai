from scontovae_train import *


#  Arguments
ontology_folder = "./projects/scontovae/data/GO/"

############

ontobj = create_ontobj(ontology_folder, "go-basic.obo", "hgnc_goterm_mapping.txt", "GO")

adata = preprocess("organoid.h5ad") # An h5ad file Converted from a Seurat .RDS file 

# Train/test split
train_adata, test_adata = split_adata(adata, 0.6)

# Integrate adata and ontobj
train_adata = integration(train_adata, ontobj)
test_adata = integration(test_adata, ontobj)

print(type(train_adata))
print(dir(train_adata))
print(type(test_adata))
print(dir(test_adata))
