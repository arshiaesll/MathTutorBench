from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig, AutoModelForVision2Seq
import torch
from datasets import load_dataset
import os
import subprocess

print("Libraries Loaded")
print("CUDA Available: ", torch.cuda.is_available())
print("CUDA Device count: ", torch.cuda.device_count())
print("World size:", os.environ.get("WORLD_SIZE"))
print("Rank:", os.environ.get("RANK"))
print("Local rank:", os.environ.get("LOCAL_RANK"))

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: ", torch.cuda.get_device_name(i))


class ModelManager():

    def __init__(self, base_model="microsoft/Phi-3.5-vision-instruct", fine_tuned_path = ""):
        
        # base_model = "microsoft/phi-3-mini-4k-instruct"
        self.base_model = base_model

        cfg = AutoConfig.from_pretrained(self.base_model, trust_remote_code = True)
        self.processor = AutoProcessor.from_pretrained(
            base_model, trust_remote_code = True)

        if cfg.model_type == "qwen2_5_vl":
            self.model = AutoModelForVision2Seq.from_pretrained(self.base_model, torch_dtype = torch.float16)

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model if fine_tuned_path == "" else fine_tuned_path,
                # The new version has dtype instead of torch_dtype
                _attn_implementation = "eager", # Not using flash attention
                torch_dtype = torch.float16,
                trust_remote_code = True,
                )
        self.processor.tokenizer.eos_token = "<|end|>"
        self.processor.tokenizer.pad_token = "<|end|>"
        self.model.config.eos_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|end|>")
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def freeze_LLM_part(self):
        # Freezing the LLM part
        for name, param in self.model.named_parameters():
            if "model.layers" in name or "lm_head" in name:
                param.requires_grad = False

        trainable = 0
        frozen = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable+= param.numel()
            else:
                frozen += param.numel()
                 
        print(f"Trainable params: {trainable:,}")
        print(f"Frozen params:   {frozen:,}")  

    def _prepare_for_inference(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device}")
        self.model.to(self.device)


    def format_messages(self, prompt, messages):
        formatted_msgs = []
        formatted_msgs.append({"role": "user", "content": prompt})
        # print("The prompt: ", prompt)
        # print("Messages: ", messages)

        return formatted_msgs

    # images is a list of image, prompt is str

    def run_inference_qwen(self, prompt = "", messages = []):
        pass
    
    def run_inference(self, prompt = "", images = [], messages = []):
        
        image_str_lst = [f"<|image_{i}|>" for i in range(len(images))]
        image_str = "\n".join(image_str_lst)
        # print("In the inference prompt ", prompt)

        if messages:
            pass
        else:
            messages = messages = [
                {"role": "user", "content": prompt + image_str}
            ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # print(prompt)
        inputs = self.processor(
            prompt,
            images if len(images) > 0 else None
        ).to(self.device)
        print("Number of images: ", len(images))
        with torch.inference_mode():
            # Using KV Cache for generation
            output = self.model.generate(
                **inputs,
                max_new_tokens = 200,
                eos_token_id = self.model.config.eos_token_id,
                pad_token_id = self.model.config.eos_token_id,
            )

        output = output[:, inputs["input_ids"].shape[1] :]
        clean_ids = output.masked_fill(output == -1, self.processor.tokenizer.eos_token_id)
        response = self.processor.tokenizer.decode(clean_ids[0], skip_special_tokens = True)

        # print(response)
        return response

    def load_dataset(self, dataset_name = "AI4Math/MathVista", split="testmini"):
        self.dataset_name = dataset_name
        dataset = load_dataset(dataset_name, split=split)
        self.dataset = dataset

    def benchmark(self, get_inputs, compare_outputs, dataset_name="AI4Math/MathVista"):
        
        self.load_dataset(dataset_name)
        self._prepare_for_inference()

        correct = 0
        total = 0
        for row in self.dataset:
            total+=1
            # Getting the image and the prompt from the dataset
            images, prompt = get_inputs(row)
            pred = self.run_inference(images=images, prompt=prompt).strip()
            correct+= compare_outputs(row, pred) 
            print("Accuracy: ", correct / total)

        return correct / total


