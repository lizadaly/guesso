"""
Guess the age of an image either from a CSV file of image information (default) or by URL.

Usage:
  guess.py
  guess.py <url>

Options:
  -h --help  Show this screen.
"""
import logging
from io import BytesIO

from docopt import docopt
import requests
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import pandas as pd
from PIL import Image

log = logging.getLogger()
logging.basicConfig(level=logging.INFO)

MODEL_NAME = "../ai-experiments/rijksdata/artbot-resnet-50.pth"

transform = transforms.Compose([
    transforms.Scale(300),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

model = models.resnet50(num_classes=1).cuda()
model.load_state_dict(torch.load(MODEL_NAME))
model.eval()


def make_guess(img, year=None):
    """Expects a PIL image, the model, and an optional year """
    tensor = transform(img)
    inp = Variable(tensor, volatile=True).cuda()
    inp = inp.unsqueeze(0)
    output = model(inp)
    output = output.squeeze()
    pred = int(output.data[0] * 100)
    if year:
        diff = abs(pred - year)
    else:
        diff = None
    return pred, diff, year, tensor


def predict_by_url(url):
    """Generate a prediction via a URL"""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    pred, diff, year, tensor = make_guess(img)
    return pred, img


if __name__ == '__main__':

    args = docopt(__doc__, version="guess.py 1.0")
    output = ""
    if args['<url>']:
        log.info("Getting image from URL %s", args['<url>'])
        pred = predict_by_url(args['<url>'])

    else:
        # Read the test data file that was created in `train.py`
        test = pd.read_csv('test-after-split.csv')
        pd.DataFrame.hist(test, 'year')

        log.info("Running test data with %d records...", len(test))
        count = 0
        avg_diff = 0
        out = []
        for f in test.itertuples():
            with Image.open(f.path).convert('RGB') as img:
                pred, diff, year, tensor = make_guess(img, year=f.year)
                avg_diff += diff
                count += 1
                out.append([year, diff, pred, f.path])

        results = pd.DataFrame(out, columns=['year', 'diff', 'pred', 'path'])
        results = results.set_index('year')
        results.to_csv('results.csv')
