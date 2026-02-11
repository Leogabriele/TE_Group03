class GGUFExporter:
    @staticmethod
    def export(model, tokenizer, path="hardened_model", quant="q4_k_m"):
        print(f"💾 Exporting to GGUF format: {quant}")
        model.save_pretrained_gguf(path, tokenizer, quantization_method=quant)