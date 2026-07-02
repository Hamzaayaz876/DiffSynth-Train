import numpy as np 
from skimage import io as img_io
from utils.word_dataset import WordLineDataset
from utils.auxilary_functions import image_resize_PIL, centered_PIL
from PIL import Image, ImageOps
import json
import os
import string

class BullingerDataset(WordLineDataset):
    def __init__(self, basefolder, subset, segmentation_level, fixed_size,  tokenizer, text_encoder, feat_extractor, transforms, args):
        super().__init__(basefolder, subset, segmentation_level, fixed_size, tokenizer, text_encoder, feat_extractor, transforms, args)
        self.setname = 'Bullinger'
        self.word_path = self.basefolder
        self.line_path = '{}/lines'.format(self.basefolder, self.setname)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.feat_extractor = feat_extractor
        self.args = args
        super().__finalize__()

    def generate_multiple_crops(img, num_crops=4, crop_size=(200, 50)):
        crops = []
        for _ in range(num_crops):
            max_x = img.size[0] - crop_size[0]
            max_y = img.size[1] - crop_size[1]
            if max_x <= 0 or max_y <= 0:  # Ensuring the crop size is smaller than the image
                # If the image is too small to be cropped, resize the original image instead
                resized_img = img.resize((crop_size[0], crop_size[1]))
                crops.append(resized_img)
            else:
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                crop = img.crop((x, y, x + crop_size[0], y + crop_size[1]))
                crops.append(crop)
        return crops
    
    
    def main_loader(self, subset, segmentation_level) -> list:
        def gather_iam_info(
            basefolder="",
            subset=""
            ):
            """
            Walk basefolder/<writer_folder> and collect:
            - image path
            - transcription (parsed from filename)
            - writer_id (remapped to 0..N-1)

            Returns:
                info_list = [(img_path, transcription, writer_id), ...]
            """
            subset_root = os.path.join(basefolder, subset)
            if not os.path.isdir(subset_root):
                raise ValueError(f"Custom dataset subset path not found: {subset_root}")

            # --------------------------------------------------
            # 1) Collect writer folders
            # --------------------------------------------------
            writers = sorted([
                w for w in os.listdir(subset_root)
                if os.path.isdir(os.path.join(subset_root, w))
            ])

            if len(writers) == 0:
                raise RuntimeError("No writer folders found in dataset!")

            # --------------------------------------------------
            # 2) Map writer folder → contiguous ID
            # --------------------------------------------------
            writer_to_id = {int(w): i for i, w in enumerate(writers)}

            print("Number of writers2:", len(writer_to_id))
            print("Writer ID mapping (first 5):", list(writer_to_id.items())[:5])

            # --------------------------------------------------
            # 3) Gather samples
            # --------------------------------------------------
            info_list = []

            for writer in writers:
                wdir = os.path.join(subset_root, writer)
                writer_id = writer_to_id[int(writer)]  # ✅ contiguous label

                for fname in sorted(os.listdir(wdir)):
                    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
                        continue

                    img_path = os.path.join(wdir, fname)

                    # transcription from filename
                    base = os.path.splitext(fname)[0]
                    if "_" in base:
                        text = base.split("_")[-1]
                    else:
                        text = base

                    text = text.replace("-", " ").strip()

                    info_list.append((img_path, text, writer_id))
            if len(info_list) == 0:
                raise RuntimeError("No images found in dataset!")

            return info_list

        info = gather_iam_info(basefolder=self.basefolder, subset=self.subset)
        print("INFO LENGTH:", len(info))
        print("FIRST 5 INFO SAMPLES:")
        for x in info[:5]:
            print(x)

        self.num_classes = len(set(writer for _, _, writer in info))
        data = []
        widths = []
        padded_imgs = 0
        padded_data = []
        character_classes = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
        for i, (img_path, transcr, writer_name) in enumerate(info):
            
            # transform iam transcriptions
            transcr = transcr.replace(" ", "")
            # "We 'll" -> "We'll"
            special_cases  = ["s", "d", "ll", "m", "ve", "t", "re"]
            # lower-case 
            for cc in special_cases:
                transcr = transcr.replace("|\'" + cc, "\'" + cc)
                transcr = transcr.replace("|\'" + cc.upper(), "\'" + cc.upper())

            transcr = transcr.replace("|", " ")
            
            if i % 1000 == 0:
                print('imgs: [{}/{} ({:.0f}%)]'.format(i, len(info), 100. * i / len(info)))
              
            try:
                #img = Image.open(img_path + '.png').convert('RGB') #.convert('L')
                img = Image.open(img_path).convert('RGB')

                #if the transcription is in stopwords
                if transcr in string.punctuation:
                    img = centered_PIL(img, (64, 256), border_value=255.0)
                
                else:
                    (img_width, img_height) = img.size
                    #resize image to height 64 keeping aspect ratio
                    img = img.resize((int(img_width * 64 / img_height), 64))
                    (img_width, img_height) = img.size
                    
                    if img_width < 256:
                        outImg = ImageOps.pad(img, size=(256, 64), color= "white")#, centering=(0,0)) uncommment to pad right
                        img = outImg
                    
                    else:
                        #reduce image until width is smaller than 256
                        while img_width > 256:
                            img = image_resize_PIL(img, width=img_width-20)
                            (img_width, img_height) = img.size
                        img = centered_PIL(img, (64, 256), border_value=255.0)

                '''
                img_padded = False
                if subset == 'train' and writer_name!=12:
                    # Create a new image by concatenating the original image with itself
                    padded_image = Image.new('RGB', (img.width * 2, img.height))
                    padded_image.paste(img, (0, 0))
                    padded_image.paste(img, (img.width, 0))
                    #padded_image.save('selfpadded.png')
                    # Construct the new image path with "_padded" added to the filename 
                    img_padded = True
                    wr_padded = writer_name
                    transcr_pad = transcr*2
                    padded_imgs += 1
                '''
            except:
               continue
            
            data += [(img, transcr, writer_name, img_path)]
            '''
            if img_padded:
                
                padded_data.append((padded_image, transcr_pad, wr_padded, img_path))
            
            #padded_data += [(padded_image, transcr*2, writer_name, img_path)]
            img_padded = False
            '''  

        print('len data', len(data))
        
        #merge data and padded_data
        data_full = data #+ padded_data
        print('len data_full', len(data_full))
        
        return data_full
