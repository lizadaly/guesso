"""
Guess the age of an image either from a CSV file of image information (default) or by URL.

Usage:
  guess.py <model_name> [--csv=<csv_file>]
  guess.py <model_name> [--url=<url>]

Options:
  -h --help  Show this screen.
  --url=<url>  Accept a URL as input
  --csv=<csv_file>  Accept a CSV file of year,path-to-file and outputs to RESULTS_FILE
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

transform = transforms.Compose([
    transforms.Scale(300),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

RESULTS_FILE = 'results.csv'

def make_guess(img, model, year=None):
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


def predict_by_url(url, model):
    """Generate a prediction via a URL"""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    pred, diff, year, tensor = make_guess(img, model)
    return pred, img


if __name__ == '__main__':

    args = docopt(__doc__, version="guess.py 1.0")
    model_name = args['<model_name>']
    model = models.resnet152(num_classes=1).cuda()
    model.load_state_dict(torch.load(model_name))
    model.eval()

    if args['--url']:
        log.info("Getting image from URL %s", args['--url'])
        pred, img = predict_by_url(args['--url'], model)
        print(pred)

    elif args['--csv']:
        # Read the test data file that was created in `train.py`
        test = pd.read_csv(args['--csv'])
        log.info("Running test data with %d records...", len(test))
        count = 0
        avg_diff = 0
        out = []
        for f in test.itertuples():
            with Image.open(f.path).convert('RGB') as img:
                pred, diff, year, tensor = make_guess(img, model, year=f.year)
                avg_diff += diff
                count += 1
                out.append([year, diff, pred, f.path])

        results = pd.DataFrame(out, columns=['year', 'diff', 'pred', 'path'])
        results = results.set_index('year')
        results.to_csv(RESULTS_FILE)
