from main import *
import pandas as pd

if __name__ == "__main__":
    svd = SVD_compress_image("mydog.jpg", rgb = True)
    image_compress, iter, psnr, cr, ss, mae, mse, sigma_value, norm = svd.find_best_compress()

    data = {"k": [i for i in range(len(norm))],
            "psnr": psnr,
            "cr": cr,
            "ss": ss,
            "mae":mae,
            "mse":mse,
            "sigma_value":sigma_value[:len(norm)],
            "2_norm":norm}

    df = pd.DataFrame(data)
    df.to_csv("result_image/svd_data", index=False, sep="\t")

    plt.plot(psnr)
    plt.title("PSNR")
    plt.xlabel("k")
    plt.savefig("result_image/psnr.jpg")
    plt.close()

    plt.plot(cr)
    plt.title("Compress Ratio")
    plt.xlabel("k")
    plt.savefig("result_image/cr.jpg")
    plt.close()

    plt.plot(ss)
    plt.title("Space Saving Rate")
    plt.xlabel("k")
    plt.savefig("result_image/ss.jpg")
    plt.close()

    plt.plot(mae)
    plt.title("MAE")
    plt.xlabel("k")
    plt.savefig("result_image/mae.jpg")
    plt.close()

    plt.plot(mse)
    plt.title("MSE")
    plt.xlabel("k")
    plt.savefig("result_image/mse.jpg")
    plt.close()

    plt.plot(sigma_value[:len(norm)])
    plt.plot(norm)
    plt.title("sigma_value and 2_norm_image_compress")
    plt.xlabel("k")
    plt.legend(["sigma_value", "2_norm"], loc="upper right")
    plt.savefig("result_image/sigma_value and 2_norm_image_compress.jpg")
    plt.close()

    plt.plot(psnr)
    plt.plot(cr)
    plt.title("the PSNR and Compress ratio")
    plt.xlabel("k")
    plt.legend(["PSNR","Compress ratio"],loc="upper right")
    plt.savefig("result_image/PSNR && Compress ratio.jpg")
    plt.close()

    psnr = [x-30 for x in psnr]
    plt.plot(psnr)
    plt.plot(cr)
    plt.title("the PSNR and Compress ratio")
    plt.xlabel("k")
    plt.legend(["PSNR", "Compress ratio"], loc="upper right")
    plt.savefig("result_image/PSNR && Compress Ratio.png")
    plt.close()







