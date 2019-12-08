import os
import random
import numpy as np
import errno
from PIL import Image
from tqdm import tqdm

def compose_images(foreground_paths, background_path):
    rnd_line_num = len(foreground_paths)

    # Make sure the background path is valid and open the image
    assert os.path.exists(background_path), 'image path does not exist: {}'.format(background_path)
    assert os.path.splitext(background_path)[1].lower() in ['.png', '.jpg', 'jpeg'], \
        'background must be a .png or .jpg file: {}'.format(background_path)
    background = Image.open(background_path)
    background = background.convert('RGBA')
    bw, bh = background.size

    # Rotate the foreground
    # angle_degrees = random.randint(0, 359)
    # foreground = foreground.rotate(angle_degrees, resample=Image.BICUBIC, expand=True)

    # Scale the foreground
    # scale = random.random() * .5 + .5  # Pick something between .5 and 1
    # new_size = (int(foreground.size[0] * scale), int(foreground.size[1] * scale))
    # foreground = foreground.resize(new_size, resample=Image.BICUBIC)

    # Add any other transformations here...
    new_foreground = Image.new('RGBA', background.size, color=(0, 0, 0, 0))
    new_alpha_mask = Image.new('L', background.size, color=0)
    bboxes = []
    for line_i in range(rnd_line_num) :
        foreground_path = foreground_paths[line_i]
        # Make sure the foreground path is valid and open the image
        assert os.path.exists(foreground_path), 'image path does not exist: {}'.format(foreground_path)
        assert os.path.splitext(foreground_path)[1].lower() == '.png', 'foreground must be a .png file'
        foreground = Image.open(foreground_path)
        fw, fh = foreground.size
        ratio = min(bw/fw, (bh/rnd_line_num)/fh/2.0, 1.0)
        #print('ratio : ', ratio)
        foreground = foreground.resize((int(fw*ratio), int(fh*ratio)), Image.BICUBIC)
        fw, fh = foreground.size
        #print('foresize : ', fw, fh)

        foreground = foreground.convert("RGBA")
        datas = foreground.getdata()
        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        foreground.putdata(newData)
        foreground_alpha = np.array(foreground.getchannel(3))
        assert np.any(foreground_alpha == 0), 'foreground needs to have some transparency: {}'.format(foreground_path)

        # Choose a random x,y position for the foreground
        max_xy_position = (bw - fw, int(bh/rnd_line_num) - fh)
        assert max_xy_position[0] >= 0 and max_xy_position[1] >= 0, \
            'foreground {} is to big for the background {}'.format(foreground_path, background_path)
        paste_position = (random.randint(0, max_xy_position[0]), int(bh*(line_i/rnd_line_num)) + random.randint(0, max_xy_position[1]))

        # Create a new foreground image as large as the background and paste it on top
        new_foreground.paste(foreground, paste_position)

        # Extract the alpha channel from the foreground and paste it into a new image the size of the background
        alpha_mask = foreground.getchannel(3)
        new_alpha_mask.paste(alpha_mask, paste_position)


        px = paste_position[0]
        py = paste_position[1]
        bboxes.append([px,py, px+fw,py, px+fw,py+fh, px,py+fh])

    composite = Image.composite(new_foreground, background, new_alpha_mask)
    return composite, bboxes

if __name__ == '__main__' :
    ap.add_argument("-f", "--from", type=int, default=1,
                    help="min number of line")
    ap.add_argument("-t", "--to", type=int, default=10,
                    help="max number of line")
    ap.add_argument("-n", "--num", type=int, default=1000,
                    help="number of genererated images")

    args = vars(ap.parse_args())
    line_min = args["from"]
    line_max = args["to"]
    case_num = args["num"]
    # Get lists of foreground and background image paths
    dataset_dir = os.path.dirname(__file__)
    backgrounds_dir = os.path.join(dataset_dir, 'backgrounds')
    foregrounds_dir = os.path.join(dataset_dir, 'foregrounds')
    backgrounds = [os.path.join(backgrounds_dir, file_name) for file_name in os.listdir(backgrounds_dir)]
    foregrounds = [os.path.join(foregrounds_dir, file_name) for file_name in os.listdir(foregrounds_dir)]

    # Create an output directory
    output_image_dir = os.path.join(dataset_dir, 'generated_image')
    output_text_dir = os.path.join(dataset_dir, 'generated_text')

    try:
        os.mkdir(output_image_dir)
        os.mkdir(output_text_dir)

    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    # Create a list to keep track of images and mask annotations
    txt_lines = []

    # Generate 5 new images
    for i in tqdm(range(case_num)):
        rnd_line_num = random.randint(line_min, line_max)
        foreground_paths = [random.choice(foregrounds) for fore_i in range(rnd_line_num)]
        #print('leng   : ' , len(foreground_paths))
        background_path = random.choice(backgrounds)
        composite, bboxes = compose_images(foreground_paths, background_path)

        composite_path = os.path.join(output_image_dir, 'img_{0:d}.png'.format(i+1))
        composite.save(composite_path)

        strbboxes = [map(str, bbox) for bbox in bboxes]
        txt_lines.append(['img_{0:d}'.format(i+1), strbboxes])

    # Output text
    for txtname, bboxes in txt_lines:
        txtpath = os.path.join(output_text_dir, txtname+'.txt')
        f = open(txtpath, 'w')
        for bbox in bboxes :
            f.write(', '.join(bbox)+ ', ###'+os.linesep)
        f.close()


#    sample_image_path = os.path.join(output_image_dir, txt_lines[0][0]+'.png')
#    sample_image = Image.open(sample_image_path)
#    sample_image.show()
