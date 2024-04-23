# This is one way I’ve done animated GIFs with matplotlib. May not be the best or only way, but it worked at the time…
 
# Let `fig` be a matplotlib Figure. Then you can use this snippet to convert it to an RGBA numpy array:
 
# ```python
# # https://stackoverflow.com/a/61443397
# ios = io.BytesIO()
# fig.savefig(ios, format='raw') # RGBA
# ios.seek(0)
# w, h = fig.canvas.get_width_height()
# return np.reshape(np.frombuffer(
#     ios.getvalue(), dtype=np.uint8), (int(h), int(w), 4))[:,:,0:4]
# ```
 
# You can then create a list `imgs` of RGBA numpy arrays and create an animated GIF `file` with:
 
# ```python
# def render_gif(imgs, file, frame_duration=0.25):
#     with imageio.get_writer(file, mode='I', duration=frame_duration) as writer:
#         for img in imgs: writer.append_data(img)
# ```

import matplotlib.pyplot as plt
import numpy as np
import io
import imageio

def render_gif(imgs, file, frame_duration=0.25):
    with imageio.get_writer(file, mode='I', duration=frame_duration) as writer:
        for img in imgs:
            writer.append_data(img)

if __name__ == "__main__":
    fig = plt.figure()

    # Create a list to hold the generated images
    imgs = []

    # Create a simple animation by updating the plot over time
    frames = 100
    for i in range(frames):
        plt.clf()  # Clear the previous plot
        x = np.linspace(0, 2*np.pi, 100)
        y = np.sin(x + i * 0.1)  # Vary the phase of the sine wave
        plt.plot(x, y)
        ios = io.BytesIO()
        fig.savefig(ios, format='raw')  # RGBA
        ios.seek(0)
        w, h = fig.canvas.get_width_height()
        img = np.reshape(np.frombuffer(ios.getvalue(), dtype=np.uint8), (int(h), int(w), 4))[:, :, 0:4]
        imgs.append(img)

    # Render the animation as a GIF
    render_gif(imgs, 'animation.gif', frame_duration=0.1)
