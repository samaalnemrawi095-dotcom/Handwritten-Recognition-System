import os
import torch
import torch.nn.functional as F

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageChops, ImageOps
from torchvision import transforms

from model import Model
from train import SAVE_MODEL_PATH
from letter_model import LetterModel


app = Flask(__name__)

SAVE_LETTER_MODEL_PATH = "./checkpoint/letter_model.pth"

predict_digit_model = None
predict_letter_model = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/DigitRecognition", methods=["POST"])
def predict_digit():
    img = Image.open(request.files["img"]).convert("L")
    mode = request.form.get("mode", "one")

    res_json = {
        "pred": 0,
        "probs": [0] * 10,
        "multi": "",
        "sum": 0
    }

    if mode == "one":
        res = predict_digit_model(img)
        res_json["pred"] = int(res.argmax())
        res_json["probs"] = (res * 100).tolist()

    else:
        digit_images = split_characters(img)

        digits = []
        probs = [0] * 10

        for digit_img in digit_images:
            res = predict_digit_model(digit_img)
            digit = int(res.argmax())
            digits.append(str(digit))
            probs = (res * 100).tolist()

        multi_number = "".join(digits)

        res_json["multi"] = multi_number
        res_json["pred"] = int(digits[-1]) if digits else 0
        res_json["probs"] = probs

        if mode == "sum":
            res_json["sum"] = sum(int(d) for d in digits)

    return jsonify(res_json)


@app.route("/LetterRecognition", methods=["POST"])
def predict_letter():
    img = Image.open(request.files["img"]).convert("L")
    mode = request.form.get("mode", "one")

    res_json = {
        "letter": "",
        "word": "",
        "probs": [0] * 26
    }

    if mode == "one":
        res = predict_letter_model(img)
        letter_index = int(res.argmax())

        res_json["letter"] = predict_letter_model.letters[letter_index]
        res_json["probs"] = (res * 100).tolist()

    else:
        letter_images = split_characters(img)

        letters = []
        probs = [0] * 26

        for letter_img in letter_images:
            res = predict_letter_model(letter_img)
            letter_index = int(res.argmax())
            letters.append(predict_letter_model.letters[letter_index])
            probs = (res * 100).tolist()

        word = "".join(letters)

        res_json["word"] = word
        res_json["letter"] = letters[-1] if letters else ""
        res_json["probs"] = probs

    return jsonify(res_json)


def split_characters(img):
    img = ImageOps.invert(img)
    bbox = img.getbbox()

    if bbox is None:
        return []

    img = img.crop(bbox)

    pixels = img.load()
    w, h = img.size

    columns = []

    for x in range(w):
        has_pixel = False

        for y in range(h):
            if pixels[x, y] > 30:
                has_pixel = True
                break

        columns.append(has_pixel)

    parts = []
    in_char = False
    start = 0

    for i, has_pixel in enumerate(columns):
        if has_pixel and not in_char:
            start = i
            in_char = True

        elif not has_pixel and in_char:
            end = i

            if end - start > 5:
                parts.append((start, end))

            in_char = False

    if in_char:
        parts.append((start, w))

    character_images = []

    for start, end in parts:
        char_img = img.crop((start, 0, end, h))
        char_img = ImageOps.invert(char_img)
        character_images.append(char_img)

    return character_images


class PredictDigit:
    def __init__(self):
        self.device = torch.device("cpu")

        self.model = Model().to(self.device)
        self.model.load_state_dict(
            torch.load(SAVE_MODEL_PATH, map_location=self.device)
        )

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def _centering_img(self, img):
        bbox = img.getbbox()

        if bbox is None:
            return img

        left, top, right, bottom = bbox
        w, h = img.size[:2]

        shift_x = (left + (right - left) // 2) - w // 2
        shift_y = (top + (bottom - top) // 2) - h // 2

        return ImageChops.offset(img, -shift_x, -shift_y)

    def __call__(self, img):
        img = ImageOps.invert(img)
        # img = self._centering_img(img)
        img = img.resize((28, 28), Image.BICUBIC) 

        tensor = self.transform(img)
        tensor = tensor.unsqueeze(0).to(self.device)

        self.model.eval()

        with torch.no_grad():
            preds = self.model(tensor)
            preds = preds.detach().cpu().numpy()[0]

        return preds


class PredictLetter:
    def __init__(self):
        self.device = torch.device("cpu")
        self.letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        self.model = LetterModel().to(self.device)
        self.model.load_state_dict(
            torch.load(SAVE_LETTER_MODEL_PATH, map_location=self.device)
        )

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def _centering_img(self, img):
        bbox = img.getbbox()

        if bbox is None:
            return img

        left, top, right, bottom = bbox
        w, h = img.size[:2]

        shift_x = (left + (right - left) // 2) - w // 2
        shift_y = (top + (bottom - top) // 2) - h // 2

        return ImageChops.offset(img, -shift_x, -shift_y)

    def __call__(self, img):
        img = ImageOps.invert(img)
        img = self._centering_img(img)
        img = img.resize((28, 28), Image.BICUBIC)

        tensor = self.transform(img)
        tensor = tensor.unsqueeze(0).to(self.device)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1)
            probs = probs.detach().cpu().numpy()[0]

        return probs


if __name__ == "__main__":
    assert os.path.exists(SAVE_MODEL_PATH), "Digit model not found"
    assert os.path.exists(SAVE_LETTER_MODEL_PATH), "Letter model not found"

    predict_digit_model = PredictDigit()
    predict_letter_model = PredictLetter()

    app.run(host="0.0.0.0", port=5001)