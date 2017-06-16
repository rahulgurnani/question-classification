# question-classification

This repository contains code for classification of questions in categories like what, when, affirmative question etc.

After installing everything, execute :

`python question_classifier.py`

Pop up would come for selecting training file (data/labelleddata.txt ) and test file.

Finally test_output.txt would be generated as the output file.

# Installation Instructions

It would be helpful installing miniconda as it will make further installations.

Install spacy using 
`conda install spacy`

Next you need to dowload english models for spacy using

`python -m spacy download en`

You can find the details of installation [here](https://spacy.io/docs/usage/).

Install python-tk, on linux machines:
`sudo apt-get install python-tk`

Pandas, Numpy, sklearn also need to be installed.

`conda install pandas`

`conda install numpy`

`conda install sklearn`
