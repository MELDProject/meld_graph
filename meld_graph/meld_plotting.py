import matplotlib.pyplot as plt
import os
import subprocess
import numpy as np
from PIL import Image, ImageChops
from PIL import ImageFont
from PIL import ImageDraw
from meld_graph.meld_cohort import MeldCohort, MeldSubject
import matplotlib_surface_plotting.matplotlib_surface_plotting as msp
from meld_graph.paths import MELD_PARAMS_PATH
import nibabel as nb

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    else:
        return im
    
def rotate90(im):
    return im.transpose(method=Image.ROTATE_270)

def plot_single_subject(data_to_plots, lesion, feature_names=None, out_filename="tmp.png"):
    """create a grid of flatmap plots for a single subject"""
    # load in meshes
    flat = nb.load(os.path.join(MELD_PARAMS_PATH, "fsaverage_sym", "surf", "lh.full.patch.flat.gii"))

    vertices, faces = flat.darrays[0].data, flat.darrays[1].data
    cortex = MeldCohort().cortex_label
    cortex_bin = np.zeros(len(vertices)).astype(bool)
    cortex_bin[cortex] = 1
    # round up to get the square grid size
    gridsize = np.ceil(np.sqrt(len(data_to_plots))).astype(int)
    ims = np.zeros((gridsize, gridsize), dtype=object)
    random = np.random.choice(100000)
    for k, data_to_plot in enumerate(data_to_plots):
        msp.plot_surf(
            vertices,
            faces,
            data_to_plot,
            flat_map=True,
            base_size=10,
            mask=~cortex_bin,
            pvals=np.ones_like(cortex_bin),
            parcel=lesion,
            vmin=np.percentile(data_to_plot[cortex_bin], 1),
            vmax=np.percentile(data_to_plot[cortex_bin], 99),
            cmap="viridis",
            colorbar=False,
            filename=out_filename,
        )
        plt.close()
#        subprocess.call(f"convert {out_filename} -trim ./tmp{random}1.png", shell=True)
 #       subprocess.call(f"convert ./tmp{random}1.png -rotate 90 {out_filename}", shell=True)
  #      os.remove(f"./tmp{random}1.png")
        im = Image.open(out_filename)
        im = trim(im)
        im = rotate90(im)
        im = im.convert("RGBA")
        #fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeSansBold.ttf", 25)
        fnt = ImageFont.load_default()
        f_name = ""
        if feature_names is not None:
            if k == 0:
                f_name = feature_names[k]
                base = np.array(im.convert("RGBA"))
            else:
                f_name = feature_names[k][35:-1]
        draw = ImageDraw.Draw(im)
        draw.text((100, 0), f_name, (255, 0, 0), font=fnt)
        arr_im = np.array(im.convert("RGBA"))
        s0 = np.min([base.shape[0], arr_im.shape[0]])
        s1 = np.min([base.shape[1], arr_im.shape[1]])
        base[:s0, :s1, :3] = arr_im[:s0, :s1, :3]

        # make transparent white
        # cropped[cropped[:,:,3]==0]=255
        base = base[:, :, :3]
        ims[k // gridsize, k % gridsize] = base.copy()

    rows = np.zeros(1 + k // gridsize, dtype=object)
    for j in np.arange(1 + k // gridsize):
        try:
            rows[j] = np.hstack(ims[j])
        except ValueError:
            ims[j, k % gridsize + 1] = np.ones_like(base) * 255
            ims[j, k % gridsize + 2] = np.ones_like(base) * 255
            rows[j] = np.hstack(ims[j])
    grid_ims = np.vstack(rows)
    im = Image.fromarray(grid_ims)
    im.save(out_filename)