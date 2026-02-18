import torch


def main() -> None:
	print(f"PyTorch version: {torch.__version__}")
	print(f"CUDA disponível: {torch.cuda.is_available()}")
	print(f"CUDA runtime (torch.version.cuda): {torch.version.cuda}")
	print(f"cuDNN disponível: {torch.backends.cudnn.is_available()}")
	print(f"cuDNN version: {torch.backends.cudnn.version()}")

	device_count = torch.cuda.device_count()
	print(f"Quantidade de GPUs detectadas: {device_count}")

if __name__ == "__main__":
	main()
