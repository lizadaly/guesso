# Guesso
## Training a convolutional neural network to guess the age of fine art paintings and drawings

<img src="https://github.com/lizadaly/guesso/blob/master/images/ex1.png?raw=true">


Using PyTorch, I took a [pretrained ResNet50 model](http://pytorch.org/docs/torchvision/models.html) and further trained it on a collection of 40,000 images of works of art from the openly licensed collections at [Rijksmuseum](https://rijksmuseum.github.io/) and [The Metropolitan Museum of Art](https://github.com/metmuseum/openaccess). 

## Detailed performance graphs
Please see the <a href="https://github.com/lizadaly/guesso/blob/master/evaluate.ipynb">Jupyter Notebook</a> in this repository for a complete walkthrough of the performance of this model.

I've never worked with convolutional neural networks before, so take this as an example of a beginner's exploration rather than any sort of best practice. PyTorch is relatively new software and I've found it useful to read other people's code, as there isn't much of it out in the world yet.

Trained model can be downloaded at: https://s3.amazonaws.com/worldwritable/nn-models/guesso-resnet-50.pth


<img src="https://github.com/lizadaly/guesso/blob/master/images/ex3.png?raw=true">

