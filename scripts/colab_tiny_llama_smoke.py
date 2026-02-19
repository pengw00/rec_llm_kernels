import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"

    tok = AutoTokenizer.from_pretrained(model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        .cuda()
        .eval()
    )

    inp = tok("hello", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model(**inp, use_cache=True)

    print("logits:", tuple(out.logits.shape))
    print("past_key_values layers:", len(out.past_key_values))


if __name__ == "__main__":
    main()

