import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
# Assuming these dependencies are available from the previous files provided by the user
from models.audio_encoder_config import AudioEncoderConfig 
from models.audio_encoder import AudioEncoderModel 


class BartCaptionModelV2(nn.Module):
    """
    V2 of the BART Caption Model designed for structured output.
    The .generate method is customized to return the top K beam search results 
    along with their scores for detailed reasoning.
    """

    def __init__(self, config):
        super(BartCaptionModelV2, self).__init__()

        self.config = config

        # --- Encoder Setup (Unchanged) ---
        encoder_config = AudioEncoderConfig(**config["audio_encoder_args"],
                                            audio_args=config["audio_args"])
        self.encoder = AudioEncoderModel(encoder_config)

        # --- Decoder Setup (Unchanged) ---
        decoder_name = config["text_decoder_args"]["name"]
        decoder_pretrained = config["text_decoder_args"]["pretrained"]
        if decoder_pretrained:
            self.decoder = BartForConditionalGeneration.from_pretrained(decoder_name)
            self.tokenizer = BartTokenizer.from_pretrained(decoder_name)
        else:
            bart_config = BartConfig.from_pretrained(decoder_name)
            self.tokenizer = BartTokenizer.from_pretrained(decoder_name)
            self.decoder = BartForConditionalGeneration.from_config(bart_config)

        self.enc_to_dec_proj = nn.Linear(encoder_config.hidden_size, self.decoder.config.hidden_size)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward_encoder(self, audios):
        """Processes audio through the HTSAT encoder and projects the embeddings."""
        outputs = self.encoder(audios)
        outputs = self.enc_to_dec_proj(outputs.last_hidden_state)
        return outputs

    # --- Only the generate method is focused on here for the new feature ---
    def generate(self,
                 samples,
                 num_beams=3, # Fixed at 3 as requested
                 max_length=50,
                 min_length=2,
                 repetition_penalty=1.0,
                 num_return_sequences=3 # Fixed at 3 as requested
                 ):
        
        self.eval() 
        with torch.no_grad():
            audio_embs = self.forward_encoder(samples)

            # 1. Prepare Encoder Outputs (Audio Embeddings)
            encoder_outputs = self.decoder.model.encoder(
                input_ids=None,
                attention_mask=None,
                inputs_embeds=audio_embs,
                return_dict=True
            )

            # 2. Prepare initial decoder input
            input_ids = torch.zeros((encoder_outputs['last_hidden_state'].size(0), 1)).long().to(self.device)
            input_ids[:, 0] = self.decoder.config.decoder_start_token_id
            decoder_attention_mask = torch.ones((encoder_outputs['last_hidden_state'].size(0), 1)).long().to(self.device)

            # 3. Execute Beam Search Generation
            # Crucial: return_dict_in_generate=True and output_scores=True are needed for reasoning
            outputs = self.decoder.generate(
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                # Arguments to return scoring information:
                return_dict_in_generate=True,
                output_scores=True,
                num_return_sequences=num_return_sequences
            )

            # 4. Extract sequences and their scores
            sequences = outputs.sequences
            sequence_scores = outputs.sequences_scores.cpu().tolist()

            # 5. Convert IDs to captions
            captions = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
            
            # 6. Combine and sort results (highest score = best caption)
            results = sorted(
                zip(captions, sequence_scores), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            return results

# Note: The other utility methods (shift_tokens_right, forward_decoder, forward) 
# required for training/full functionality are omitted for brevity but assumed 
# to be present in the original BartCaptionModel.