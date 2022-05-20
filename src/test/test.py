numbers = '0123456789'
words = 'abcdefghijklmnopqrstuvwxyz-'

def solution(files: list):
    split_files = [split_information(file) for file in files]

def split_information(file):
    tail_start_idx = 0
    for i, string in enumerate(file):
        if string in numbers:
            num_start_idx = i
        elif string 

files = ["img12.png", "img10.png", "img02.png", "img1.png", "IMG01.GIF", "img2.JPG"]

