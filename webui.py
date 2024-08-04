# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav, speed_change

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

max_val = 0.8
def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech

inference_mode_list = ['Pre-trained voices', '3s instant cloning', 'Multi-lingual generation', 'Natural language control']
instruct_dict = {'Pre-trained voices': '1. Select the pre-trained voice\n2. Click the Generate Audio button',
                 '3s instant cloning': '1. Select the audio file, or record audio. Note that it does not exceed 30 seconds. If both are provided, the audio file will be selected first\n2. Enter the prompt\n3. Click the Generate Audio button',
                 'Multi-lingual generation': '1. Select the audio file, or record audio. Note that it does not exceed 30 seconds. If both are provided, the audio file will be selected first\n2. Enter the prompt\n3. Click the Generate Audio button',
                 'Natural language control': '1. Select the pre-trained voice\n2. Enter the instruct text\n3. Click the Generate Audio button'}
                 
def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]

def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text, seed, speed_factor):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['Natural language control']:
        if cosyvoice.frontend.instruct is False:
            gr.Warning('You are using the natural language control mode. The {} model does not support this mode. Please use the iic/CosyVoice-300M-Instruct model.'.format(args.model_dir))
            return (target_sr, default_data)
        if instruct_text == '':
            gr.Warning('You are using natural language control mode, please enter instruct text')
            return (target_sr, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info('You are using natural language control mode, prompt audio/prompt text will be ignored')
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['Multi-lingual generation']:
        if cosyvoice.frontend.instruct is True:
            gr.Warning('You are using the cross-language reproduction mode. The {} model does not support this mode. Please use the iic/CosyVoice-300M model.'.format(args.model_dir))
            return (target_sr, default_data)
        if instruct_text != '':
            gr.Info('You are using cross-language replication mode, instruct text will be ignored')
        if prompt_wav is None:
            gr.Warning('You are using cross-language reproduction mode, please provide prompt audio')
            return (target_sr, default_data)
        gr.Info('You are using cross-language replication mode. Please ensure that the synthesized text and prompt text are in different languages.')
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s instant cloning', 'Multi-lingual generation']:
        if prompt_wav is None:
            gr.Warning('The prompt audio is empty. Did you forget to enter the prompt audio?')
            return (target_sr, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('Prompt audio sampling rate {} is lower than {}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            return (target_sr, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['Pre-trained voices']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('You are using pre-trained tone mode, prompt text/prompt audio/instruct text will be ignored!')
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['3s instant cloning']:
        if prompt_text == '':
            gr.Warning('The prompt text is empty, did you forget to enter the prompt text?')
            return (target_sr, default_data)
        if instruct_text != '':
            gr.Info('You are using the 3s speed reproduction mode, and the pre-training sounds/instruct text will be ignored!')

    if mode_checkbox_group == 'Pre-trained voices':
        logging.info('get sft inference request')
        set_all_random_seed(seed)
        output = cosyvoice.inference_sft(tts_text, sft_dropdown)
    elif mode_checkbox_group == '3s instant cloning':
        logging.info('get zero_shot inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    elif mode_checkbox_group == 'Multi-lingual generation':
        logging.info('get cross_lingual inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    else:
        logging.info('get instruct inference request')
        set_all_random_seed(seed)
        output = cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text)

    if speed_factor != 1.0:
        try:
            audio_data, sample_rate = speed_change(output["tts_speech"], target_sr, str(speed_factor))
            audio_data = audio_data.numpy().flatten()
        except Exception as e:
            print(f"Failed to change speed of audio: \n{e}")
    else:
        audio_data = output['tts_speech'].numpy().flatten()

    return (target_sr, audio_data)

def main():
    with gr.Blocks() as demo:
        gr.Markdown("### Code base [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) Pre-trained model [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) [CosyVoice -300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice- 300M-SFT)")
        gr.Markdown("#### Please enter the text to be synthesized, select the inference mode, and follow the prompts.")

        tts_text = gr.Textbox(label="Enter text to generate", lines=1, value="I am a newly launched large generative speech model by the speech team of Tongyi Lab, which provides comfortable and natural speech synthesis capabilities.")
        speed_factor = gr.Slider(minimum=0.25, maximum=4, step=0.05, label="Speech rate adjustment", value=1.0, interactive=True)
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='Select inference mode', value=inference_mode_list[0])
            instruction_text = gr.Text(label="Steps", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='Select a pre-trained voice', value=sft_spk[0], scale=0.25)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="Seed")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='Select the prompt audio file and note that the sampling rate is not less than 16khz')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='Record prompt audio file')
        prompt_text = gr.Textbox(label="Enter prompt text", lines=1, placeholder="Please enter the prompt text, which must be consistent with the prompt audio content. Automatic recognition is not supported for the time being....", value='')
        instruct_text = gr.Textbox(label="Enter instruct text", lines=1, placeholder="Please enter instruct text.", value='')

        generate_button = gr.Button("Generate audio")

        audio_output = gr.Audio(label="Output")

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text, seed, speed_factor],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice-300M',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = CosyVoice(args.model_dir)
    sft_spk = cosyvoice.list_avaliable_spks()
    prompt_sr, target_sr = 16000, 22050
    default_data = np.zeros(target_sr)
    main()
