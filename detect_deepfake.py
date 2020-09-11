import torch
import sys
from utils import *
from models.DeepFakeDetectModel_1 import *
import cv2




def main():
    print_banner()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = DeepFakeDetectModel_1().to(device)


if __name__ == '__main__':
    main()
