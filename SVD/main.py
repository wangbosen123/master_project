import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import math
import time


class SVD_compress_image():
    def __init__(self, path, rgb):
        self.path = path
        self.rgb = rgb


    def deal_data(self):
        if self.rgb:
            image = cv2.imread(self.path)
            image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            return image_rgb

        image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        return image


    def compress_process(self,u,s,v,k):
        ak = np.zeros((len(u), len(v)))

        for i in range(k):
            ak += s[i] * np.dot(u[:, i:i + 1], v[i:i + 1, :])

        return np.rint(ak).astype('int')


    def SVD_compress(self,k):
        image = self.deal_data()
        if self.rgb:
            ur, sr, vr = np.linalg.svd(image[:, :, 0], full_matrices=1)
            ug, sg, vg = np.linalg.svd(image[:, :, 1], full_matrices=1)
            ub, sb, vb = np.linalg.svd(image[:, :, 2], full_matrices=1)
            ar = self.compress_process(ur, sr, vr, k)
            ag = self.compress_process(ug, sg, vg, k)
            ab = self.compress_process(ub, sb, vb, k)
            ak = np.stack((ar, ag, ab),axis=-1)
            return ak
        else:
            u, s, v = np.linalg.svd(image, full_matrices=1)
            ak = self.compress_process(u, s, v, k)

            return ak


    def measure(self,image,image_compress,k):
        plt.imshow(image_compress)
        plt.savefig(f"result_image_compress/image_compress{k}.jpg")
        plt.close()
        image_size = os.path.getsize(self.path)
        image_compress_size = os.path.getsize(f"result_image_compress/image_compress{k}.jpg")
        if self.rgb:
            ur, sr, vr = np.linalg.svd(image[:, :, 0], full_matrices=1)
            ug, sg, vg = np.linalg.svd(image[:, :, 1], full_matrices=1)
            ub, sb, vb = np.linalg.svd(image[:, :, 2], full_matrices=1)
            sigma = np.add(np.add(sr,sg),sb)
        else:
            u, sigma, v = np.linalg.svd(image, full_matrices=1)

        mae = np.mean(abs(image - image_compress))
        mse = np.mean(np.square(image - image_compress))
        if self.rgb:
            two_norm = 0
            for i in range(3):
                two_norm += np.linalg.norm(image[:, :, i] - image_compress[:, :, i], ord=2)
        else:
            two_norm = np.linalg.norm(image - image_compress, ord=2)

        psnr = 10 * np.log10(255**2 / mse)
        cr = image_size / image_compress_size
        ss = 1 - 1/cr

        return psnr, cr, ss, mae, mse, sigma, two_norm

    def find_best_compress(self):
        image = self.deal_data()
        psnr_total = []
        cr_total = []
        ss_total = []
        mae_total = []
        mse_total = []
        norm = []
        for k in range(min(image.shape[0], image.shape[1])):
            start = time.time()
            image_compress = self.SVD_compress(k)
            psnr, cr, ss, mae, mse, sigma_value, two_norm = self.measure(image, image_compress, k)
            psnr_total.append(psnr)
            cr_total.append(cr)
            ss_total.append(ss)
            mae_total.append(mae)
            mse_total.append(mse)
            norm.append(two_norm)

            print("______________________")
            print(f"the k : {k+1}")
            print(f"psnr is {psnr_total[-1]}")
            print(f"cr is {cr_total[-1]}")
            print(f"ss is {ss_total[-1]}")
            print(f"mae is {mae_total[-1]}")
            print(f"mse is {mse_total[-1]}")
            print(f"the 2_norm is {norm[-1]}")
            print(f"sigma_value is {sigma_value[k]}")
            print(f"the spend time %s second" %(time.time() - start ))
            if psnr_total[-1] > 30 and cr_total[-1] < psnr_total[-1]-30:
                break

        self.draw_sample_graph()

        return image_compress, k, psnr_total, cr_total, ss_total, mae_total, mse_total, sigma_value,norm



    def draw_sample_graph(self, path="result_image"):
        t=10
        ax,fig = plt.subplots(figsize=(30,8))
        for i in range(10):
            plt.subplot(2,10,i+1)
            plt.title(f"k = {i+1}")
            plt.axis("off")
            if self.rgb:
                plt.imshow(self.SVD_compress(i))
            plt.imshow(self.SVD_compress(i), cmap="gray")

            plt.subplot(2,10,i+11)
            plt.title(f"k = {i+t}")
            plt.axis("off")
            if self.rgb:
                plt.imshow(self.SVD_compress(i+t))
            plt.imshow(self.SVD_compress(i+t), cmap="gray")
            t+=20
        plt.savefig(path + "/" + "result.jpg")
        plt.close()







if __name__ == '__main__':
    svd = SVD_compress_image("mydog.jpg", rgb = False)
    svd.SVD_compress(10)
    # # (1108,1147)
    # pic = svd.deal_data()
    # plt.imshow(pic,cmap="gray")
    # plt.show()
    # ak = svd.SVD_compress(40)
    # plt.imshow(ak,cmap="gray")
    # plt.show()

    # psnr, cr, ss, mae, mse = svd.measure(pic, ak,1)
    # print(psnr, cr, ss, mae, mse)







