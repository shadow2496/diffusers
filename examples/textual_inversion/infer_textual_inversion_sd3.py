import argparse
from glob import glob
import os
import shutil

from diffusers import StableDiffusion3Pipeline
import numpy as np
from PIL import Image
import safetensors
import torch
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm


def load_model(text_encoder, tokenizer, save_path, resize_token_embeddings=True):
    st = safetensors.torch.load_file(save_path)
    placeholder_tokens = list(st.keys())
    print(placeholder_tokens)

    placeholder_token_ids = []
    for placeholder_token in placeholder_tokens:
        _ = tokenizer.add_tokens(placeholder_token)
        placeholder_token_ids.append(tokenizer.convert_tokens_to_ids(placeholder_token))

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    if resize_token_embeddings:
        text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    for placeholder_token, placeholder_token_id in zip(placeholder_tokens, placeholder_token_ids):
        token_embeds[placeholder_token_id] = st[placeholder_token]

    return ' '.join(placeholder_tokens)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='stable-diffusion-3-medium-diffusers-b1148b4/', help='path to saved model')
    parser.add_argument('--lora_ckpt', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument(
        "--from_file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument('--prompt', type=str)

    parser.add_argument('-b', '--batch_size', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--save_grid', action='store_true')

    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    parser.add_argument('--n_steps', type=int, default=28)  # TODO: 왜 default값이 28인지 확인
    parser.add_argument('--scale', type=float, default=7.0)  # TODO: 왜 default값이 7.0인지 확인
    parser.add_argument('--latents_checkpoint', type=str, default=None)

    args = parser.parse_args()
    return args


def save_new_image(args, dirnames, new_filename):
    new_img = Image.new('RGB', (1024 * args.n_samples, 1024 * len(dirnames)))
    for i, dirname in enumerate(dirnames):
        filenames = [f for f in sorted(os.listdir(os.path.join(args.save_dir, dirname))) if f.endswith('.png')]
        for j, filename in enumerate(filenames):
            img = Image.open(os.path.join(args.save_dir, dirname, filename))
            new_img.paste(img, (1024 * j, 1024 * i))
            if 'image000' in filename:
                print('{} is opened'.format(os.path.join(dirname, filename)))

    new_img.save('{}_{}'.format(args.save_dir[:-1], new_filename))
    print('{} is saved'.format(new_filename))
    print()


def main():
    args = parse_args()
    # TODO: safety_checker 역할을 하는 module이 SD3에는 없는지 확인
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_dir, torch_dtype=torch.float16, revision=args.revision
    ).to("cuda")
    # pipe.load_lora_weights(args.lora_ckpt)
    placeholder_token = load_model(pipe.text_encoder, pipe.tokenizer, args.lora_ckpt)
    load_model(pipe.text_encoder_2, pipe.tokenizer_2, args.lora_ckpt.replace('embeds', 'embeds_2'))
    load_model(pipe.text_encoder_3, pipe.tokenizer_3, args.lora_ckpt.replace('embeds', 'embeds_3'), resize_token_embeddings=False)

    if args.seed:
        print(f"Global seed set to {args.seed}")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if not args.from_file:
        prompt = args.prompt
        assert prompt is not None
        data = [prompt.lower()]
    else:
        print(f"reading prompts from {args.from_file}")
        with open(args.from_file, "r") as f:
            data = f.read().splitlines()
            data = [d.strip("\n").split("\t")[0] for d in data]
            data = [d.lower() for d in data]

    suffix = ''
    if args.scale != 7.0:
        suffix = '{}_cfg{:04.1f}'.format(suffix, args.scale)
    if args.n_steps != 28:
        suffix = '{}_step{:03d}'.format(suffix, args.n_steps)
    suffix = '{}_s{:04d}_{:03d}'.format(suffix, args.seed, args.n_samples)
    if args.latents_checkpoint:
        suffix = '{}_val'.format(suffix)

    if args.latents_checkpoint:
        latents = torch.load(args.latents_checkpoint)

    is_too_long = False
    for idx, prompt in tqdm(enumerate(data), position=1, desc='data'):
        text = prompt.replace('<new1> ', '').replace(' <new1>', '')
        prompt = prompt.replace('<new1>', placeholder_token)

        if not args.from_file:
            save_dir = os.path.join(args.save_dir, '{}{}'.format(prompt.replace(' ', '-'), suffix))
        else:
            save_dir = os.path.join(args.save_dir, '{:02d}_{}{}'.format(idx, prompt.replace(' ', '-'), suffix))
        while True:
            try:
                os.makedirs(save_dir, exist_ok=True)
                break
            except:
                is_too_long = True
                split = save_dir.split('/')
                if not args.from_file:
                    split[-1] = split[-1][1:]
                else:
                    split[-1] = '{:02d}_{}'.format(idx, split[-1][4:])
                save_dir = '/'.join(split)

        if args.save_grid:
            images_tensor = []
        for i in range(0, args.n_samples, args.batch_size):
            prompts = [prompt for _ in range(min(args.batch_size, args.n_samples - i))]
            if args.latents_checkpoint and i + args.batch_size <= latents.size(0):
                images = pipe(
                    prompt=prompts,
                    negative_prompt="",
                    num_inference_steps=args.n_steps,
                    height=1024,
                    width=1024,
                    guidance_scale=args.scale,
                    latents=latents[i:i + args.batch_size],
                ).images
            else:
                images = pipe(
                    prompt=prompts,
                    negative_prompt="",
                    num_inference_steps=args.n_steps,
                    height=1024,
                    width=1024,
                    guidance_scale=args.scale,
                ).images

            for j, image in enumerate(images):
                image.save(os.path.join(save_dir, 'image{:03d}_{}_.png'.format(i + j, text)))
                if args.save_grid:
                    images_tensor.append(transforms.ToTensor()(image))

        if args.save_grid:
            grid = torch.stack(images_tensor, 0)
            grid = make_grid(grid, nrow=args.n_samples)
            transforms.ToPILImage()(grid).save(os.path.join(save_dir, 'all.jpg'))

    if args.from_file:
        dirnames = [path.split('/')[-2] for path in sorted(glob(os.path.join(args.save_dir, '??_*{}/'.format(suffix))))]
        save_new_image(args, dirnames, 'all_00-{:02d}{}.jpg'.format(len(dirnames) - 1, suffix))

    if is_too_long:
        split = args.save_dir.split('/')
        split[-2] = '{}_too_long'.format(split[-2])
        shutil.move(args.save_dir, '/'.join(split))


if __name__ == '__main__':
    main()
