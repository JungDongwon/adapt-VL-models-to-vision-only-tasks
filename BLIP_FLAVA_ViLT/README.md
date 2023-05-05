This folder contains training scripts and exploratory notebooks for the BLIP, FLAVA, and ViLT models. The training scripts allow for selecting each of the 4 adaptations.

Training logs are present in the `logs` subfolder.

`BLIP-openai-evaluator.py` contains the code for using the OpenAI API to translate the outputs of BLIP to one of the potential class labels of CIFAR-100. Note that use of this script expects a `.env` file to be present in this directory containing your OpenAI API key. Also note that the script allows for using GPT-4 or GPT-3.5 models.

`BLIP_Final_Results` contains the results of OpenAI translation of outputs to labels.

`Old_Files` archives other scripts and notebooks that were used when working with these models.
