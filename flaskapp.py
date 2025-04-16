from pathlib import Path

import flask
import torch

from models.classifier import AttentionClassifierConfig, AttentionClassifier
from models.classifier import LSTMClassifierConfig, LSTMClassifier
from models.classifier import ClassifierConfig, TransformerClassifier

from common import read_yaml
from utils import load_tokenizer


def load_checkpoint(filename: str | Path) -> torch.nn.Module:
    ckpt = torch.load(filename, map_location='cpu')

    cfg = ckpt['config']
    print(cfg)
    if 'epoch' in ckpt:
        print('model saved on epoch {}'.format(ckpt['epoch']))

    if isinstance(cfg, AttentionClassifierConfig):
        model = AttentionClassifier(cfg)
    elif isinstance(cfg, LSTMClassifierConfig):
        model = LSTMClassifier(cfg)
    elif isinstance(cfg, ClassifierConfig):
        model = TransformerClassifier(cfg)

    model.load_state_dict(ckpt['state_dict'])

    return model


def clean_token(t: str):
    if ord(t[0]) == 288:
        t = t[1:]

    return t


class EndpointAction:
    def __init__(self, action, mimetype=None):
        self.action = action
        self.mimetype = mimetype if mimetype is not None else 'text/plain'

    def __call__(self, **kwargs):
        retval = self.action(**kwargs)
        response = flask.Response(retval, status=200, headers={}, mimetype=self.mimetype)
        return response


class ClassifierApp:
    def __init__(self, model, tokenizer, port=58000, device: torch.device | None = None):
        self.app = flask.Flask(__name__)

        self.device = device if device is not None else torch.device('cpu')

        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.model.eval()
        self.port = port

        self.add_endpoint(endpoint='/', endpoint_name='front', handler=self.front, mimetype='text/html', methods=['GET', 'POST'])

    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None, mimetype=None, methods=None):
        methods = methods if methods is not None else ['GET']
        self.app.add_url_rule(endpoint, endpoint_name, EndpointAction(handler, mimetype=mimetype), methods=methods)

    def run(self):
        self.app.run(host='0.0.0.0', port=self.port)

    @torch.inference_mode()
    def classify(self, prompt):
        tokens = torch.as_tensor(self.tokenizer.encode(prompt).ids, dtype=torch.int64, device=self.device).unsqueeze(0)
        y_pred, scores = self.model(tokens)
        decoded = [clean_token(self.tokenizer.id_to_token(t.item())) for t in tokens[0][1:-1]]

        m = 'audio' if torch.argmax(y_pred[0], dim=-1).item() == 0 else 'image'
        p = torch.softmax(y_pred[0], dim=-1).cpu().numpy()

        s1 = scores[0][0, 0, :]
        #s2 = scores[0][0, :, 0]

        return m, p, s1[1:-1].cpu().numpy(), decoded

    def front(self, **kwargs):
        if flask.request.method == 'GET':
            return flask.render_template('index.html')
        else:
            # analyze
            prompt = flask.request.form.get('prompt')
            res, probs, scores, tokens = self.classify(prompt)

            #tokens = '[' + ', '.join(map(lambda s: f'"{s}"', tokens)) + ']'
            return flask.render_template('results.html', result=res, probs=probs, tokens=tokens, scores=list(scores))


def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)

    tokenizer_dir = cfg.get('tokenizer_dir')
    tokenizer = load_tokenizer(tokenizer_dir)

    model = load_checkpoint(cfg.get('model_path'))

    app = ClassifierApp(model=model, tokenizer=tokenizer, port=58000)
    app.run()


if __name__ == '__main__':
    main()


