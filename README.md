# Guesso
## Training a convolutional neural network to guess the age of fine art paintings and drawings

<img src="https://github.com/lizadaly/guesso/blob/master/images/ex1.png?raw=true">

Using PyTorch, I took a [pretrained ResNet152 model](http://pytorch.org/docs/torchvision/models.html) and further trained it on a collection of 40,000 images of works of art from the openly licensed collections at [Rijksmuseum](https://rijksmuseum.github.io/) and [The Metropolitan Museum of Art](https://github.com/metmuseum/openaccess). On average, it guesses the age of a work within 66 years.

This project is referenced in my article, [AI Literacy: How artificial intelligence classifies and predicts our world](https://worldwritable.com/ai-literacy-what-artificial-intelligence-can-do-part-2-cbca0fc75a93).

## Detailed performance graphs
Please see the <a href="https://github.com/lizadaly/guesso/blob/master/evaluate.ipynb">Jupyter Notebook</a> in this repository for a complete walkthrough of the performance of this model.

I've never worked with convolutional neural networks before, so take this as an example of a beginner's exploration rather than any sort of best practice. PyTorch is relatively new software and I've found it useful to read other people's code, as there isn't much of it out in the world yet.

## Trained model
* <a href="http://lizadaly.com/projects/guesso/guesso-resnet-152.pth.gz">ResNet 152</a> (avg error +/- 66 years, 222MB)

## How to use this code

You can either use one of the pre-trained models above, or train your own convnet on a corpus of artwork
where you have metadata as well as imagery (and appropriate licensing!).

The `metadata` directory contains two CSV files mapping year-of-work to IDs from Rijksmuseum and the Met. You can use those
IDs to acquire the full images from the providers. If you want to train on another task besides year, get the
full metadata sets from the providers and use those.

If you use one of the pre-trained models, `guess.py` can accept either an image by URL or a CSV in the
form of true year, path-to-image-file, like so:

```
year,path
1803,/home/liza/data/rijksmuseum-images/214/RP-T-FM-241.jpeg
1886,/home/liza/data/met-images/435/435649.jpg
1480,/home/liza/data/met-images/337/337494.jpg
```

(The `train.py` script will automatically generate a `test-after-split.csv` file in this format, by pulling out 10% of the training data as a validation set.)

If a URL is passed, the script will return the prediction and a PIL object. The CSV form
will output its predictions to another CSV, as:

```
year,diff,pred,path
1803,6,1797,/home/liza/data/rijksmuseum-images/214/RP-T-FM-241.jpeg
1886,304,1582,/home/liza/data/met-images/435/435649.jpg
1480,158,1638,/home/liza/data/met-images/337/337494.jpg
1700,159,1859,/home/liza/data/met-images/365/365519.jpg
```

suitable for analysis, as in `evaluate.ipynb`

Happy guessing!

Liza Daly
https://twitter.com/liza

<img src="https://github.com/lizadaly/guesso/blob/master/images/ex3.png?raw=true">
