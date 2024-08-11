import os
from typing import Sequence
from dataclasses import dataclass
from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from dacite import from_dict
from transformers import GPT2Tokenizer
from Helpers.loaders import ModelSaverReader
from xlstm.xlstm.utils import WeightDecayOptimGroupMixin
from xlstm.xlstm.components.init import small_init_init_
from xlstm.xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig

# Configuration settings from environment variables
PORT = int(os.getenv('PORT', 8000))
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False') == 'True'
MODEL_PATH = os.getenv('MODEL_PATH', './Models/model1_bin_final.pth')
SCALER_PATH = os.getenv('SCALER_PATH', 'fitted_scaler.joblib')

# Initialize the Flask application
app = Flask(__name__)

# Load configuration
cfg = OmegaConf.load('./params_app.yaml')

@dataclass
class xLSTMLMModelConfig(xLSTMBlockStackConfig):
    vocab_size: int = -1
    tie_weights: bool = True
    weight_decay_on_embedding: bool = True
    add_embedding_dropout: bool = True

class xLSTMLMModel(WeightDecayOptimGroupMixin, nn.Module):
    config_class = xLSTMLMModelConfig

    def __init__(self, config: xLSTMLMModelConfig, **kwargs):
        super().__init__()
        self.config = config

        self.xlstm_block_stack = xLSTMBlockStack(config=config)
        self.token_embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim)
        self.emb_dropout = nn.Dropout(config.dropout) if config.add_embedding_dropout else nn.Identity()

        self.lm_head = nn.Linear(
            in_features=config.embedding_dim,
            out_features=config.vocab_size,
            bias=False,
        )
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    def reset_parameters(self):
        self.xlstm_block_stack.reset_parameters()
        small_init_init_(self.token_embedding.weight, dim=self.config.embedding_dim)
        if not self.config.tie_weights:
            small_init_init_(self.lm_head.weight, dim=self.config.embedding_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(idx)
        x = self.emb_dropout(x)
        x = self.xlstm_block_stack(x)
        logits = self.lm_head(x)
        return logits

    def step(
        self, idx: torch.Tensor, state: dict[str, dict[str, tuple[torch.Tensor, ...]]] = None, **kwargs
    ) -> tuple[torch.Tensor, dict[str, dict[str, tuple[torch.Tensor, ...]]]]:
        x = self.token_embedding(idx)
        x = self.emb_dropout(x)
        x, state = self.xlstm_block_stack.step(x, state=state, **kwargs)
        logits = self.lm_head(x)
        return logits, state

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        weight_decay, no_weight_decay = super()._create_weight_decay_optim_groups(**kwargs)
        weight_decay = list(weight_decay)
        removed = 0
        for idx in range(len(weight_decay)):
            if weight_decay[idx - removed] is self.token_embedding.weight:
                weight_decay.pop(idx - removed)
                removed += 1
        weight_decay = tuple(weight_decay)
        if self.config.weight_decay_on_embedding:
            weight_decay += (self.token_embedding.weight,)
        else:
            no_weight_decay += (self.token_embedding.weight,)
        return weight_decay, no_weight_decay

# Define binary classification model
class xLSTMLMModelBinary(nn.Module):
    def __init__(self, config, pretrained_model=None):
        super(xLSTMLMModelBinary, self).__init__()
        if pretrained_model is None:
            self.pretrained_model = xLSTMLMModel(config)
        else:
            self.pretrained_model = pretrained_model
        self.fc = nn.Linear(config.embedding_dim, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        hidden_states = self.pretrained_model.token_embedding(input_ids)
        hidden_states = self.pretrained_model.xlstm_block_stack(hidden_states)
        pooled_outputs = hidden_states.mean(dim=1)
        pooled_outputs = self.dropout(pooled_outputs)
        logits = self.fc(pooled_outputs)
        return logits.squeeze(-1)

# Load the pre-trained weights into the binary classification model
def load_pretrained_weights(pretrained_model, binary_model):
    pretrained_dict = pretrained_model.state_dict()
    model_dict = binary_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
    model_dict.update(pretrained_dict)
    binary_model.load_state_dict(model_dict)
    return binary_model

class HateSpeechDetector:
    def __init__(self, model, tokenizer, context_length, device):
        self.model = model
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.device = device

    def predict(self, tweet):
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer.encode_plus(
                tweet,
                add_special_tokens=True,
                max_length=self.context_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = inputs['input_ids'].to(self.device)
            outputs = self.model(input_ids)
            prediction = torch.sigmoid(outputs).item()
            is_hate = prediction >= 0.3
            return is_hate

# Access the schedul dictionary directly
schedul = {
    1: cfg.model.schedul['first'],
    int(cfg.training.num_steps * (1/8)): cfg.model.schedul['quarter'],
    int(cfg.training.num_steps * (1/4)): cfg.model.schedul['half'],
    int(cfg.training.num_steps * (1/2)): cfg.model.schedul['three_quarters']
}
final_context_length = schedul[max(schedul.keys())]
cfg.model.context_length = final_context_length

# Load the model using ModelSaverReader
device = torch.device('cpu')
model_saver_reader = ModelSaverReader('./Models')
model_bin_final_10k = model_saver_reader.load_model(
    xLSTMLMModelBinary, 
    "model1_bin_final", 
    from_dict(xLSTMLMModelConfig, OmegaConf.to_container(cfg.model, resolve=True)),
    map_location=device  # Ensure the model is loaded on the CPU
).to(device)
model_bin_final_10k.eval()


# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Initialize the detector
detector = HateSpeechDetector(model_bin_final_10k, tokenizer, cfg.model.context_length, device)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        tweet = request.form['tweet']
        is_hate = detector.predict(tweet)
        label = 'Hate' if is_hate else 'Not Hate'
        color = 'red' if is_hate else 'green'
        return render_template_string('''
            <form method="POST">
                <textarea id="tweet" name="tweet" rows="4" cols="50" placeholder="Enter your tweet here...">{{ tweet }}</textarea><br>
                <input type="submit" value="Submit">
                <button type="button" onclick="resetForm()">Reset</button>
            </form>
            <p>Result: <span style="color:{{ color }};">{{ label }}</span></p>
            <script>
                const tweetField = document.getElementById('tweet');
                if ('{{ color }}' === 'red') {
                    let blinkCount = 0;
                    const blinkInterval = setInterval(() => {
                        tweetField.style.backgroundColor = tweetField.style.backgroundColor === 'red' ? 'white' : 'red';
                        blinkCount++;
                        if (blinkCount === 6) {
                            clearInterval(blinkInterval);
                            tweetField.style.backgroundColor = 'white';
                        }
                    }, 200);
                } else {
                    tweetField.style.backgroundColor = '{{ color }}';
                    setTimeout(() => {
                        tweetField.style.backgroundColor = 'white';
                    }, 3000);
                }
                function resetForm() {
                    tweetField.value = '';
                    tweetField.style.backgroundColor = 'white';
                }
            </script>
        ''', label=label, color=color, tweet=tweet)
    return '''
        <form method="POST">
            <textarea id="tweet" name="tweet" rows="4" cols="50" placeholder="Enter your tweet here..."></textarea><br>
            <input type="submit" value="Submit">
            <button type="button" onclick="resetForm()">Reset</button>
        </form>
        <script>
            function resetForm() {
                const tweetField = document.getElementById('tweet');
                tweetField.value = '';
                tweetField.style.backgroundColor = 'white';
            }
        </script>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    tweet = data.get('tweet', '')
    is_hate = detector.predict(tweet)
    return jsonify({'is_hate': is_hate})

# Start the Flask app
<<<<<<< HEAD
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)

=======
if __name__=='__main__':
    app.run(host='0.0.0.0', port=PORT)
>>>>>>> 0af7232 (Initial commit for new repository)
