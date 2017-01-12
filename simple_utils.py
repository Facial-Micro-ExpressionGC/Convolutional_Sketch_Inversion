import matplotlib.pyplot as plt
import numpy as np


def plot_batch(model, img_size, batch_size, sketch, color, epoch, idx, tag, nb_img=5):
    img_sketch = np.array(sketch[0:nb_img])
    img_color = np.array(color[0:nb_img])
    img_gen = model.predict(sketch, batch_size=batch_size)[0][0:nb_img]
    for i in range(nb_img):
        plt.subplot(nb_img, 3, i * 3 + 1)
        plt.imshow(img_sketch[i].reshape((img_size,img_size)), cmap='Greys_r')
        plt.axis('off')
        plt.subplot(nb_img, 3, i * 3 + 2)
        plt.imshow(img_color[i])
        plt.axis('off')
        plt.subplot(nb_img, 3, i * 3 + 3)
        plt.imshow(img_gen[i])
        plt.axis('off')
    plt.savefig("figures/%s_fig_epoch%s_idx%s.png" % (tag, epoch, idx))
    plt.clf()
    plt.close()

