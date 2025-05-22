#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import csv
import json

def parse_resolution(resolution_str):
    if "1024 x 1024" in resolution_str:
        return 1024, 1024
    elif "768 x 1360" in resolution_str:
        return 768, 1360
    elif "1360 x 768" in resolution_str:
        return 1360, 768
    elif "880 x 1168" in resolution_str:
        return 880, 1168
    elif "1168 x 880" in resolution_str:
        return 1168, 880
    elif "1248 x 832" in resolution_str:
        return 1248, 832
    elif "832 x 1248" in resolution_str:
        return 832, 1248
    else:
        return 1024, 1024  # Default fallback

class PromptLoader:
    def __init__(
            self,
            prompt_file: str,
            prompt_file_type: str,
            batch_size: int,
            num_images_per_prompt: int = 1,
            max_num_prompts: int = 0
    ):
        self.prompts = []
        self.catagories = ['Not_specified']
        self.batch_size = batch_size
        self.num_images_per_prompt = num_images_per_prompt

        if prompt_file_type == 'plain':
            self.load_prompts_plain(prompt_file, max_num_prompts)
        elif prompt_file_type == 'parti':
            self.load_prompts_parti(prompt_file, max_num_prompts)
        elif prompt_file_type == 'hpsv2':
            self.load_prompts_hpsv2(prompt_file, max_num_prompts)
        else:
            print("This operation is not supported!")

        self.current_id = 0
        self.inner_id = 0

    def __len__(self):
        return len(self.prompts) * self.num_images_per_prompt

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_id == len(self.prompts):
            raise StopIteration

        ret = {
            'prompts': [],
            'catagories': [],
            'save_names': [],
            'n_prompts': self.batch_size,
        }
        for _ in range(self.batch_size):
            if self.current_id == len(self.prompts):
                ret['prompts'].append('')
                ret['save_names'].append('')
                ret['catagories'].append('')
                ret['n_prompts'] -= 1

            else:
                prompt, catagory_id = self.prompts[self.current_id]
                ret['prompts'].append(prompt)
                ret['catagories'].append(self.catagories[catagory_id])
                ret['save_names'].append(f'{self.current_id}_{self.inner_id}')

                self.inner_id += 1
                if self.inner_id == self.num_images_per_prompt:
                    self.inner_id = 0
                    self.current_id += 1

        return ret

    def load_prompts_plain(self, file_path: str, max_num_prompts: int):
        with os.fdopen(os.open(file_path, os.O_RDONLY), "r") as f:
            for i, line in enumerate(f):
                if max_num_prompts and i == max_num_prompts:
                    break

                prompt = line.strip()
                self.prompts.append((prompt, 0))

    def load_prompts_parti(self, file_path: str, max_num_prompts: int):
        with os.fdopen(os.open(file_path, os.O_RDONLY), "r") as f:
            # Skip the first line
            next(f)
            tsv_file = csv.reader(f, delimiter="\t")
            for i, line in enumerate(tsv_file):
                if max_num_prompts and i == max_num_prompts:
                    break

                prompt = line[0]
                catagory = line[1]
                if catagory not in self.catagories:
                    self.catagories.append(catagory)

                catagory_id = self.catagories.index(catagory)
                self.prompts.append((prompt, catagory_id))

    def load_prompts_hpsv2(self, file_path: str, max_num_prompts: int):
        with open(file_path, 'r') as file:
            all_prompts = json.load(file)
        count = 0
        for style, prompts in all_prompts.items():
            for prompt in prompts:
                count += 1
                if max_num_prompts and count >= max_num_prompts:
                    break

                if style not in self.catagories:
                    self.catagories.append(style)

                catagory_id = self.catagories.index(style)
                self.prompts.append((prompt, catagory_id))