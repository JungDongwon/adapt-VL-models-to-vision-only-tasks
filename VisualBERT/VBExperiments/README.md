This subfolder contains the final scripts for producing image embeddings as input into VisualBERT, as well as training scripts/notebooks for the different adaptations for VisualBERT.

To generate embeddings, please see the notebook `get_cifar10_embeddings.ipynb` and `get_cifar100_embeddings.ipynb`. Note that the code for generating image embeddings is adapted from the huggingface repository at https://github.com/huggingface/transformers/tree/main/examples/research_projects/visual_bert .

The notebook `get_cifar100_embeddings_top_box_only.ipynb` outlines an alternate method for getting embeddings whereby only the highest scoring bounding box is used. However this was not used for our final results.

Other notebooks include the training and evaluation code for testing VisualBERT with each of our adaptation methods.
